"""PromptRegistry - the prompt registry (Phase 11).

    All prompt texts are assembled in one place. The twelve prompts split into two
    groups (the owner's ruling: "some prompts are ours to polish; nobody edits
    them"):

    - **4 editable** (`summary_system`, `summary_merge_system`,
      `summary_split_system`, `judge_system`): with `prompts_dir` set they are
      written out to files (seed-if-missing - created with the built-in default as
      the starting text) and edited by the operator without an image rebuild;
      existing operator-edited files are NEVER overwritten; an empty file = the
      built-in default; a non-empty file wins. Auto-migration (lsbdef-0006): a
      file byte-identical to a previously seeded text (seed_meta.json) or to a
      known legacy seed (LEGACY_SEEDS) is rewritten with the current canon when
      the seed version changes;
    - **8 hardwired** (`merge_user`, `split_user`, `judge_user`,
      `classifier_system`, `describe_system`, `describe_user`,
      `structure_judge_system`, `structure_judge_user`): code constants only,
      never created as files.

    The contract guard: the final `judge_system` (built-in or from a file) MUST
    contain the "DUPLICATE" and "NOT DUPLICATE" markers - the dedup verdicts are
    parsed by them (app/services/judge.py::_verdict), and a file without the
    markers would silently kill dedup with parser failures. A violation - a fatal
    `ConfigError` at registry construction (the service start aborts with a
    readable message).

    `prompts_dir` unset (None) - the file mechanics are entirely off: only the
    built-in constants work.

    The consumer - pool 4 (the summary/judge slot clients take the system prompts
    from the registry; the user templates are the registry's hardwired constants).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from app.config import MAX_NOTE_CHARS_DEFAULT, ConfigError
from app.services.decision import DECISION_KINDS, DecisionQuestion


def _build_decision_questions(
    groups: dict, max_questions: int
) -> dict[str, tuple[DecisionQuestion, ...]]:
    """Assemble and validate the decision-question groups (arch §6.4).

            The validation is fatal (`ConfigError`, the `judge_system` markers
            precedent): an unknown `kind`, an empty name/text, a `choice` without
            labels or criteria, a name duplicate in a group, more questions in a group
            than `DECISION_MAX_QUESTIONS`, a `threshold` outside 0..1. Returns
            `group -> tuple[DecisionQuestion]`.

    """
    result: dict[str, tuple[DecisionQuestion, ...]] = {}
    for group, items in groups.items():
        if not isinstance(items, (list, tuple)):
            raise ConfigError(
                f"decision questions group «{group}»: expected a list of questions"
            )
        if len(items) > max_questions:
            raise ConfigError(
                f"decision questions group «{group}»: {len(items)} questions "
                f"is more than DECISION_MAX_QUESTIONS ({max_questions})"
            )
        seen: set[str] = set()
        built: list[DecisionQuestion] = []
        for item in items:
            if not isinstance(item, dict):
                raise ConfigError(
                    f"decision questions group «{group}»: a question must be an object"
                )
            kind = item.get("kind")
            name = item.get("name")
            if kind not in DECISION_KINDS:
                raise ConfigError(
                    f"decision question «{name}»: unknown kind «{kind}» — "
                    "one of " + ", ".join(DECISION_KINDS)
                )
            if not isinstance(name, str) or not name.strip():
                raise ConfigError(
                    f"decision questions group «{group}»: empty question name"
                )
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ConfigError(f"decision question «{name}»: empty text")
            if name in seen:
                raise ConfigError(
                    f"decision questions group «{group}»: duplicate question "
                    f"name «{name}»"
                )
            seen.add(name)
            labels = tuple(str(label) for label in (item.get("labels") or ()))
            criteria = item.get("criteria")
            if kind == "choice":
                if not labels:
                    raise ConfigError(
                        f"decision question «{name}»: choice question without labels"
                    )
                if not criteria:
                    raise ConfigError(
                        f"decision question «{name}»: choice question without criteria"
                    )
            threshold = item.get("threshold")
            if threshold is not None:
                value = float(threshold)
                if not 0.0 <= value <= 1.0:
                    raise ConfigError(
                        f"decision question «{name}»: threshold {value} is out of 0..1"
                    )
            built.append(
                DecisionQuestion(
                    name=name,
                    text=text,
                    kind=kind,
                    labels=labels,
                    criteria=dict(criteria) if criteria else None,
                    threshold=float(threshold) if threshold is not None else None,
                )
            )
        result[str(group)] = tuple(built)
    return result


#--- Built-in texts -------------------------------------------------------
# Moved 1:1 from app/services/summary.py, judge.py, classifier.py,
# promotion.py (the source constants stay in place until the clients move
# to the registry - pool 4; see test_builtin_texts_match_current_constants).

# The 3 editable ones (seed-if-missing with prompts_dir set).

# summary.py::SYSTEM_PROMPT - the note summary.
# The length limit - 150 CHARACTERS (the 3.1.0 gate decision): the former
#"no more than 30 words" is gone; the prompt carries no double cap. The prompt
# is a request to the model; the guard against an undisciplined model is in
# code: the word-boundary cut at the save point (worker.process_summary_pending,
# summary.cap_summary, the settings.max_summary_chars limit).
SUMMARY_SYSTEM_PROMPT = (
    "Summarize the note in 1–2 short, dense sentences — no more than 150 "
    "characters in total. Convey the main idea so that this condensed version "
    "alone makes the text perfectly clear. No intros, quotes, or "
    "explanations. Respond in the language of the note."
)

# summary.py::MERGE_SYSTEM_PROMPT - merging duplicates (system).
# The LSBN-F-23 canon: a merge returns a LIST of self-contained parts (JSON) -
#"several self-contained notes are better than one cut short". The part
# limit - the note soft limit (settings.max_note_chars; since 3.4.1 - from
# the config, LSBN-T-26); the contract is structured so the CODE checks
# completeness, not an eyeball. The number is substituted by the builder below -
# the canon has no literal (one number - one place: the MAX_NOTE_CHARS env).
def summary_merge_system_prompt(limit: int) -> str:
    """The merge system prompt; `limit` - the note soft limit from settings."""
    return (
        "You have two versions of the same note. Merge them into one or several "
        "SELF-CONTAINED notes (parts): combine all information from both — facts, "
        "names, numbers, dates, statuses, configs and paths; state each fact once, "
        "omit repetitions; do not drop anything and do not add anything of your "
        "own. Every part must be readable on its own, without references to other "
        "parts. Use ONE part only if the merged text fits the "
        f"{limit}-character note limit; otherwise split the content into "
        "several parts, each within the same limit. Give each part its own `title`: "
        "a meaningful name of at most 5 words. Write coherently, no headings, "
        "intros, quotes, or explanations. Respond in the language of the notes. "
        "Answer with exactly one JSON object, no explanations: "
        '{"parts": [{"title": "...", "text": "..."}, ...]}'
    )


# summary.py::SPLIT_SYSTEM_PROMPT - splitting a long note (system).
# The LSBN-F-22 canon (M2): the same part-list contract as the merge, but
# the source is ONE long note; the part self-sufficiency requirement is
# strengthened by the design (the step "splitting goes through F-23"): each
# part carries ITS OWN names, numbers, dates - readable without the source
def summary_split_system_prompt(limit: int) -> str:
    """The split system prompt; `limit` - the note soft limit from settings."""
    return (
        "You are given ONE long note that exceeds the note limit. Split it into "
        "one or several SELF-CONTAINED notes (parts): preserve every fact — "
        "names, numbers, dates, statuses, configs and paths; do not drop "
        "anything and do not add anything of your own. Every part must be "
        f"readable on its own, carrying ITS OWN names, numbers and dates — no "
        "references to the source or to other parts. Parts may be up to "
        f"{limit} characters each. Give each part its own `title`: a "
        "meaningful name of at most 5 words. Write coherently, no headings, "
        "intros, quotes, or explanations. Respond in the language of the note. "
        "Answer with exactly one JSON object, no explanations: "
        '{"parts": [{"title": "...", "text": "..."}, ...]}'
    )

# and the other parts.
# judge.py::JUDGE_SYSTEM_PROMPT - deciding note sameness (DUPLICATE/NOT DUPLICATE).
JUDGE_SYSTEM_PROMPT = (
    "You check long-term memory for duplicates. Determine whether two texts "
    "are duplicates: the same thought restated in other words (matching "
    "details matter more than wording). Answer with exactly one marker: "
    "DUPLICATE or NOT DUPLICATE. No explanations."
)

# The 7 hardwired ones (constants only, never created as files).

# summary.py::MERGE_USER_TEMPLATE ({text_a}, {text_b}).
SUMMARY_MERGE_USER_TEMPLATE = "TEXT 1:\n{text_a}\n\nTEXT 2:\n{text_b}"

# summary.py::SPLIT_USER_TEMPLATE ({text}) - splitting a long note (F-22 M2).
SUMMARY_SPLIT_USER_TEMPLATE = "TEXT:\n{text}"

# judge.py::JUDGE_USER_TEMPLATE ({text_new}, {text_candidate}).
JUDGE_USER_TEMPLATE = "TEXT 1:\n{text_new}\n\nTEXT 2:\n{text_candidate}"

# classifier.py::CLASSIFY_SYSTEM_PROMPT - the placement JSON contract (§5.7).
CLASSIFIER_SYSTEM_PROMPT = (
    "You are a note classifier for hierarchical memory. Determine which "
    "section the note belongs to. Known nodes are listed in the request. "
    "Rules: if the note belongs to an existing node — return its full path "
    "as hint_path (1..3 segments, latin letters, digits, hyphens); if the "
    "note is generic and not bound to a section — return null. Answer with "
    "exactly one JSON object, no explanations: "
    '{"hint_path": "...", "confidence": 0.0} — confidence from 0 to 1, '
    "how confident you are."
)

# promotion.py::DESCRIBE_SYSTEM_PROMPT - a node description from note examples.
DESCRIBE_SYSTEM_PROMPT = (
    "You generate descriptions of sections in hierarchical memory. From the "
    "example notes, write a description of the section: what kind of notes "
    "live there. Strictly 1–2 short sentences, no lists or explanations — "
    "only the description."
)

# promotion.py::DESCRIBE_USER_TEMPLATE ({domain}, {slug}, {summaries}).
DESCRIBE_USER_TEMPLATE = (
    "New subsection: {domain}/{slug}\n\n"
    "Example notes of the section (brief summaries):\n{summaries}"
)

# promotion.py::JUDGE_SYSTEM_PROMPT - the structure judge
# (the CREATE/MERGE <path>/REJECT protocol).
STRUCTURE_JUDGE_SYSTEM_PROMPT = (
    "You are the structure judge of hierarchical memory. Evaluate the "
    "candidate for a new sub-section. Rules: (1) if the meaning of the "
    "candidate matches an existing topical node (the same theme in other "
    "words) — it is a merge, not a new node; (2) the slug and the "
    "description must be meaningful: a meaningless, junk or empty-in-meaning "
    "candidate — reject. Answer with exactly one marker, no explanations: "
    "CREATE — the candidate is new and meaningful; MERGE <path> — the "
    "candidate duplicates an existing node, for path specify ONLY the "
    "topical path from the \"Existing nodes\" list (never the candidate's "
    "path; default is the system swap, merging into it never happens); "
    "REJECT — the candidate is meaningless."
)

# promotion.py::JUDGE_USER_TEMPLATE ({domain}, {slug}, {description},
# {nodes}, {nearest}).
STRUCTURE_JUDGE_USER_TEMPLATE = (
    "Candidate: {domain}/{slug} — {description}\n\n"
    "Existing nodes:\n{nodes}\n\n"
    "Nearest by vector similarity: {nearest}"
)

# Name -> built-in text (one dictionary: both the registry properties and the seed files).
# The texts with a number (merge/split) are assembled by the builders from the
# config DEFAULT; PromptRegistry at construction re-assembles them from the
# passed settings.max_note_chars (3.4.1, LSBN-T-26: one number - one place).
BUILTIN_PROMPTS: dict[str, str] = {
    "summary_system": SUMMARY_SYSTEM_PROMPT,
    "summary_merge_system": summary_merge_system_prompt(MAX_NOTE_CHARS_DEFAULT),
    "summary_split_system": summary_split_system_prompt(MAX_NOTE_CHARS_DEFAULT),
    "judge_system": JUDGE_SYSTEM_PROMPT,
    "merge_user": SUMMARY_MERGE_USER_TEMPLATE,
    "split_user": SUMMARY_SPLIT_USER_TEMPLATE,
    "judge_user": JUDGE_USER_TEMPLATE,
    "classifier_system": CLASSIFIER_SYSTEM_PROMPT,
    "describe_system": DESCRIBE_SYSTEM_PROMPT,
    "describe_user": DESCRIBE_USER_TEMPLATE,
    "structure_judge_system": STRUCTURE_JUDGE_SYSTEM_PROMPT,
    "structure_judge_user": STRUCTURE_JUDGE_USER_TEMPLATE,
}

# Editable: only these names are written to files (the hardwired - never).
EDITABLE_PROMPTS: tuple[str, ...] = (
    "summary_system",
    "summary_merge_system",
    "summary_split_system",
    "judge_system",
)

_PROMPT_FILE_SUFFIX = ".txt"

# Version of the current prompt canon (seed stamp). Bumped on every canon
# change of the editable prompts; used for auto-migration of untouched
# legacy seeds (lsbdef-0006). The current canon is 3.4.1 (LSBN-F-22 M2: added
# the editable `summary_split_system` - splitting a long note into a list of
# self-contained parts; 3.4.0 - merging into a list of parts, LSBN-F-23; earlier
# 3.1.0 - summaries <= 150 characters, the gate decision). The stamp was set
# by the 3.4.0 release and does not change at 3.4.1: rewriting untouched
# seeds goes through the current-canon comparison (_should_migrate) - the
# in-text limit number now comes from settings, and a MAX_NOTE_CHARS change
# rewrites the seed without a stamp bump (LSBN-T-26).
SEED_VERSION = "3.4.1"

# Name of the seed-stamp sidecar file in prompts_dir: version + SHA-256 of
# the seeded content per editable prompt.
_SEED_META_FILENAME = "seed_meta.json"

# --- Legacy seeds (before the stamp was introduced) ------------------------
# RU canon of v2.1.x for the three editable prompts (extracted from git
# history, commit 227c110^, before lsb-0006). Installs that survived v2.1.x
# have these files in prompts/ without seed_meta.json; auto-migration
# recognizes them as untouched seeds and rewrites them with the current canon.
SUMMARY_SYSTEM_PROMPT_V21 = (
    "Сделай краткий пересказ заметки в 1–2 коротких и ёмких "
    "предложениях, суммарно не длиннее 30 слов. Передай главную мысль "
    "так, чтобы по этому сокращению было предельно понятно, о чём текст. "
    "Без вступлений, кавычек и пояснений. Отвечай на языке заметки."
)

# EN canon of v2.2.1 (the «no more than 30 words» wording): installs upgraded
# to 3.1.0 have this text in prompts/summary_system.txt — auto-migration
# recognizes it as an untouched seed and rewrites it with the current canon
# (≤ 150 characters, gate decision 2026-09-19).
SUMMARY_SYSTEM_PROMPT_V221 = (
    "Summarize the note in 1–2 short, dense sentences, no more than 30 "
    "words in total. Convey the main idea so that this condensed version "
    "alone makes the text perfectly clear. No intros, quotes, or "
    "explanations. Respond in the language of the note."
)

SUMMARY_MERGE_SYSTEM_PROMPT_V21 = (
    "У тебя две версии одной заметки. Сведи их в единый текст: объедини "
    "всю информацию обеих — факты, имена, числа, даты, статусы, конфиги "
    "и пути; каждый факт скажи один раз, повторяющееся опусти. Ничего "
    "не выбрасывай и не добавляй от себя. Пиши связно, без заголовков, "
    "вступлений, кавычек и пояснений. Отвечай на языке заметок."
)

JUDGE_SYSTEM_PROMPT_V21 = (
    "Ты проверяешь долговременную память на дубли. Определи, являются ли "
    "два текста дублями: одна и та же мысль, пересказанная другими словами "
    "(совпадение деталей важнее формы). Ответь строго одной отметкой: "
    "ДУБЛЬ или НЕ ДУБЛЬ. Без пояснений."
)

# Prompt name → known legacy seeds (for auto-migration of installs predating
# seed_meta.json). A file byte-identical to one of them is treated as an
# untouched legacy seed and rewritten with the current canon.
LEGACY_SEEDS: dict[str, tuple[str, ...]] = {
    "summary_system": (SUMMARY_SYSTEM_PROMPT_V21, SUMMARY_SYSTEM_PROMPT_V221),
    "summary_merge_system": (SUMMARY_MERGE_SYSTEM_PROMPT_V21,),
    "judge_system": (JUDGE_SYSTEM_PROMPT_V21,),
}


#--- built-in decision questions (LSBN-F-12, arch §6.4) -----------------------
# Not prompts but question data: a name, a wording (EN, depersonalized), a type
# and criteria. The choice criteria come in `state` (the node descriptions);
# here - the generic fallback label. A group is overridden by the file
# `decision_questions.json` in prompts_dir WHOLE (§6.4).
BUILTIN_DECISION_QUESTIONS: dict[str, tuple[dict, ...]] = {
    "judge": (
        {
            "name": "is_text_duplicate",
            "text": (
                "Is TEXT 2 a duplicate of TEXT 1 — the same thought restated "
                "in other words? Matching details matter more than wording."
            ),
            "kind": "noul",
            "criteria": {
                "true": "both texts carry the same fact with matching details",
                "false": "the texts differ in facts, details or intent",
            },
        },
    ),
    "classify": (
        {
            "name": "note_section",
            "text": (
                "Which section does the note belong to? Pick the best matching "
                "known node from the request, or the generic label when the "
                "note is not bound to any section."
            ),
            "kind": "choice",
            "labels": ("generic",),
            "criteria": {
                "generic": "the note is generic and not bound to a known section",
            },
        },
    ),
}

# The override file name of the decision-question groups (we never create it).
_DECISION_QUESTIONS_FILENAME = "decision_questions.json"


class PromptRegistry:
    """The built-in constants of all 12 prompts + the file overrides.

            Construction = the full readiness preparation: seed-if-missing (with
            `prompts_dir` set), reading the overrides and the contract check of
            `judge_system`. The validation fails with a fatal `ConfigError` - the
            service start must abort (see the module docstring).

            Access - by the prompt names (properties, as pool 3 specified): the
            editable `summary_system`, `summary_merge_system`, `summary_split_system`,
            `judge_system` and the hardwired `merge_user`, `split_user`, `judge_user`,
            `classifier_system`, `describe_system`, `describe_user`,
            `structure_judge_system`, `structure_judge_user`; the universal
            `get(name)` - for iterating over the `BUILTIN_PROMPTS` list.

    """

    def __init__(
        self,
        prompts_dir: str | os.PathLike[str] | None = None,
        *,
        decision_max_questions: int = 64,
        max_note_chars: int = MAX_NOTE_CHARS_DEFAULT,
    ) -> None:
        # A copy of the built-ins: the file overrides layer on top. The texts
        # with the note soft limit (merge/split) are re-assembled from the
        # passed value - the build's settings.max_note_chars (3.4.1,
        # LSBN-T-26: one env change updates both the prompts and the validation).
        self._texts: dict[str, str] = {
            **BUILTIN_PROMPTS,
            "summary_merge_system": summary_merge_system_prompt(max_note_chars),
            "summary_split_system": summary_split_system_prompt(max_note_chars),
        }
        # The names whose final text came from a non-empty file (the source
        # diagnostics in the validation errors).
        self._file_overrides: set[str] = set()
        self.prompts_dir: Path | None = (
            Path(prompts_dir) if prompts_dir is not None else None
        )
        # LSBN-F-12: the decision-question registry (the validation is fatal, §6.4).
        self._decision_questions: dict[str, tuple[DecisionQuestion, ...]] = (
            _build_decision_questions(BUILTIN_DECISION_QUESTIONS, decision_max_questions)
        )
        if self.prompts_dir is not None:
            self._load_files()
            self._load_decision_file(decision_max_questions)
        self._validate_judge_markers()

    #--- API (the names are the pool 3/4 contract) -------------------------------------

    @property
    def summary_system(self) -> str:
        """The note-summary system prompt (editable)."""
        return self._texts["summary_system"]

    @property
    def summary_merge_system(self) -> str:
        """The duplicate-merge system prompt (editable)."""
        return self._texts["summary_merge_system"]

    @property
    def summary_split_system(self) -> str:
        """The long-note split system prompt (editable, F-22 M2)."""
        return self._texts["summary_split_system"]

    @property
    def judge_system(self) -> str:
        """The dedup judge DUPLICATE/NOT DUPLICATE system prompt (editable)."""
        return self._texts["judge_system"]

    @property
    def merge_user(self) -> str:
        """The merge user template: {text_a}, {text_b} (hardwired)."""
        return self._texts["merge_user"]

    @property
    def split_user(self) -> str:
        """The split user template: {text} (hardwired, F-22 M2)."""
        return self._texts["split_user"]

    @property
    def judge_user(self) -> str:
        """The dedup judge user template: {text_new}, {text_candidate} (hardwired)."""
        return self._texts["judge_user"]

    @property
    def classifier_system(self) -> str:
        """The classifier system prompt, a JSON contract (hardwired)."""
        return self._texts["classifier_system"]

    @property
    def describe_system(self) -> str:
        """The node description system prompt (hardwired)."""
        return self._texts["describe_system"]

    @property
    def describe_user(self) -> str:
        """The node description user template: {domain}/{slug}, {summaries} (hardwired)."""
        return self._texts["describe_user"]

    @property
    def structure_judge_system(self) -> str:
        """The structure judge CREATE/MERGE/REJECT system prompt (hardwired)."""
        return self._texts["structure_judge_system"]

    @property
    def structure_judge_user(self) -> str:
        """The structure judge user template (hardwired)."""
        return self._texts["structure_judge_user"]

    def get(self, name: str) -> str:
        """The prompt text by name (see BUILTIN_PROMPTS); KeyError - unknown."""
        try:
            return self._texts[name]
        except KeyError:
            raise KeyError(
                f"PromptRegistry: неизвестный промпт «{name}» — доступны: "
                + ", ".join(BUILTIN_PROMPTS)
            ) from None

    @property
    def decision_questions(self) -> dict[str, tuple[DecisionQuestion, ...]]:
        """The decision-question groups (LSBN-F-12): a mapping `group -> tuple` (§6.4)."""
        return self._decision_questions

    #--- decision questions (LSBN-F-12, arch §6.4) -----------------------------

    def _load_decision_file(self, max_questions: int) -> None:
        """An optional `decision_questions.json` overrides a group whole.

                    The file is neither created nor rewritten (the `_load_files`
                    precedent: the operator's edits are untouchable); a read/format error
                    - a fatal `ConfigError`. A file group replaces the built-in one whole.

        """
        assert self.prompts_dir is not None
        path = self.prompts_dir / _DECISION_QUESTIONS_FILENAME
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(
                f"{_DECISION_QUESTIONS_FILENAME}: cannot read/parse ({exc})"
            ) from exc
        if not isinstance(data, dict):
            raise ConfigError(
                f"{_DECISION_QUESTIONS_FILENAME}: top level must be an object "
                "of question groups"
            )
        groups = dict(BUILTIN_DECISION_QUESTIONS)
        groups.update(data)
        self._decision_questions = _build_decision_questions(groups, max_questions)

    #--- the file mechanics ---------------------------------------------------

    def _file_path(self, name: str) -> Path:
        assert self.prompts_dir is not None
        return self.prompts_dir / f"{name}{_PROMPT_FILE_SUFFIX}"

    def _load_files(self) -> None:
        """Seed-if-missing for the three editable prompts + auto-migration of stale seeds.

        Existing files are NEVER overwritten if they are operator-edited.
        Auto-migration (lsbdef-0006): a file byte-identical to a previously
        seeded text (seed_meta.json) or to a known legacy seed (LEGACY_SEEDS)
        is treated as an untouched legacy seed and rewritten with the current
        canon. An operator-edited file (matching no known seed) is left
        alone — FATAL remains an honest signal. An empty (whitespace-only)
        file → built-in default; a non-empty file wins.
        """
        directory = self.prompts_dir
        assert directory is not None
        directory.mkdir(parents=True, exist_ok=True)
        meta = self._read_meta()
        for name in EDITABLE_PROMPTS:
            path = self._file_path(name)
            if not path.exists():
                # First start: create the file with the built-in default as
                # the starting text (the text is already in self._texts — no
                # re-read needed) and record the seed stamp.
                path.write_text(self._texts[name], encoding="utf-8")
                meta["files"][name] = self._hash(self._texts[name])
                continue
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                # An empty file → built-in default (as before).
                continue
            if self._should_migrate(name, content, meta):
                # Auto-migration: untouched legacy seed → current canon.
                path.write_text(self._texts[name], encoding="utf-8")
                meta["files"][name] = self._hash(self._texts[name])
                continue
            # Operator-edited / current file wins.
            self._texts[name] = content
            self._file_overrides.add(name)
        meta["seed_version"] = SEED_VERSION
        self._write_meta(meta)

    # --- seed stamp (lsbdef-0006) ------------------------------------------

    def _meta_path(self) -> Path:
        assert self.prompts_dir is not None
        return self.prompts_dir / _SEED_META_FILENAME

    def _read_meta(self) -> dict:
        """Read seed_meta.json; on absence/corruption — empty stamp."""
        path = self._meta_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("files"), dict):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return {"seed_version": "", "files": {}}

    def _write_meta(self, meta: dict) -> None:
        self._meta_path().write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _should_migrate(self, name: str, content: str, meta: dict) -> bool:
        """Whether the file needs auto-migration to the current canon.

                    True if the file is an untouched seed: either seeded by us and
                    unchanged since (the hash matches seed_meta.json) while the built-in
                    canon moved on - a bumped stamp version OR the built-in text itself
                    differs (since 3.4.1/LSBN-T-26 the in-text number comes from settings:
                    a MAX_NOTE_CHARS change alters the built-in text, and an untouched
                    seed is rewritten without a stamp-version bump) - or byte-identical to
                    a known legacy seed.

        """
        recorded = meta["files"].get(name)
        if recorded is not None and self._hash(content) == recorded:
            # File unchanged since our seed → rewrite when the current
            # canon differs (seed version bumped or the built-in text
            # moved, e.g. the note limit now comes from the config).
            return (
                meta.get("seed_version") != SEED_VERSION
                or content != self._texts[name]
            )
        # File not seeded by us (legacy) — checking known legacy seeds.
        return content in LEGACY_SEEDS.get(name, ())

    #--- validation -----------------------------------------------------------

    def _validate_judge_markers(self) -> None:
        """The final judge_system must contain the "DUPLICATE" and "NOT DUPLICATE" markers.

                    The dedup verdicts are parsed by these markers (judge._verdict: first
                    "NOT DUPLICATE", then "DUPLICATE"). A prompt without the markers would
                    give the parser no verdict at all - dedup would silently stall on
                    refusals. Fatal.

                    "DUPLICATE" must appear as a standalone occurrence (outside the "NOT
                    DUPLICATE" bundle): a prompt allowing only "NOT DUPLICATE" would
                    silently break merging in the other direction - the judge could not
                    produce a "duplicate" verdict (the same failure class as losing the
                    marker entirely).

        """
        judge = self._texts["judge_system"]
        missing = self._missing_judge_markers(judge)
        if missing:
            if "judge_system" in self._file_overrides:
                origin = f"file {self._file_path('judge_system')}"
            else:
                origin = "builtin judge_system"
            raise ConfigError(
                "judge_system prompt is missing verdict marker(s): "
                + " and ".join(missing)
                + f" (source: {origin}). Dedup verdicts are parsed by these "
                "markers (judge._verdict); without them dedup would silently "
                "stall on parser failures. Restore both markers — DUPLICATE "
                "and NOT DUPLICATE — in the text. To re-seed from the built-in "
                "default, delete the file and restart; or edit it to add both "
                "markers."
            )

    @staticmethod
    def _missing_judge_markers(text: str) -> list[str]:
        """The verdict markers missing from the prompt text.

                    "DUPLICATE" counts only as a standalone occurrence: the occurrences
                    inside "NOT DUPLICATE" are struck out. "NOT DUPLICATE" - as a whole
                    bundle.

        """
        missing: list[str] = []
        if "NOT DUPLICATE" not in text:
            missing.append("NOT DUPLICATE")
        if "DUPLICATE" not in text.replace("NOT DUPLICATE", ""):
            missing.append("DUPLICATE")
        return missing
