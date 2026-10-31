"""EmbeddingService - vectorization through the embedding LLM slot (ARCH §3.2, §5.3).

    The single outbound call: the embedding slot's embed - `POST /api/embed` at an
    ollama provider, `POST /v1/embeddings` at openai (Phase 11). FULL TEXTS are
    encoded (notes and search queries), not summaries (REQUIREMENTS §5.3).

    Phase 11: the HTTP transport moved into `LLMClient` (app/services/llm_client.py):
    one client per embedding slot - the provider `ollama` | `openai`, the per-slot
    address/model/key from Settings (SlotSpec.for_embedding). The service no
    longer keeps httpx nor assembles payloads itself: the F1 queue (ollama_slot -
    the ollama provider only), Bearer with a key set, the timeouts (connect 2 s
    LAN / 10 s cloud, read 720 s) and the response parse live inside the client.

    Key decisions:
    - The synchronous path: the services are called from `asyncio.to_thread` (the
      event loop is not occupied); `LLMClient` keeps an httpx keep-alive pool -
      no TCP handshake per request.
    - The read timeout is a fixed 720 s (SS8 sets no embedding timeout; sized for
      CPU inference: qwen3-embedding:8b on a CPU host encodes ~3.7k tokens in
      ~2 min - long texts must finish, not tear; an earlier finish returns
      earlier. Short texts stay in the milliseconds range, REQUIREMENTS §5.1).
    - One retry in the synchronous path (ARCH §3.2): transient failures only -
      the `LLMError.transient` signal (429/5xx/transport/timeout). 401/403
      (`LLMAuthError`) and other 4xx never retry; the client error text
      (including the {SLOT}_API_KEY hint on an auth refusal) is preserved.
    - The response is checked hard: the vector list length equals the input
      count, and every vector's dimension equals EMBEDDING_DIM. Garbage from a
      proxy or someone else's model does not pass further as "valid vectors"
      (the vec0 table pins the dimension - a mismatch would otherwise break the
      restart).
    - Any failure - `EmbeddingError`, and only it: the calling code decides the
      degradation itself (save -> vector_status=pending, search -> FTS-only;
      steps 3.3-3.4). No failure ever breaks a user operation (NFR-3).
    - `last_attempt_ok` (NFR-4) - the source of `/health.embedding_ok`: None -
      no attempts yet; True/False - the last attempt's outcome. The startup slot
      check (main.py) initializes it before the first real call.
    - For unit tests - a transport without the network: the constructor's
      `transport` takes an httpx.MockTransport or a handler (passed into
      LLMClient); for the service tests there is the deterministic HashEmbedder
      (tests/fakes.py, ARCH §7: hash -> vector).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import httpx

from app.config import Settings
from app.services.llm_client import LLMClient, LLMError, SlotSpec
from app.services.slot_outcome import SlotOutcome, classify_llm_error

# One retry in the synchronous path (ARCH §3.2): transient failures only -
# the `LLMError.transient` signal (429/5xx/transport/timeout); 401/403
# (LLMAuthError) and other 4xx never retry.
MAX_ATTEMPTS = 2


class EmbeddingError(RuntimeError):
    """The vectorization failed: the server is unavailable or the response is malformed.

            `outcome` - the single slot-call outcome type (No39, arch §6.4): a client
            failure - `classify_llm_error`; a response-contract violation (the count/
            dimension of the vectors) - explicitly `OFF_CONTRACT` (the default). The
            worker journals the outcome.

    """

    def __init__(
        self,
        message: str,
        *,
        outcome: SlotOutcome = SlotOutcome.OFF_CONTRACT,
    ) -> None:
        super().__init__(message)
        self.outcome = outcome


class Embedder(Protocol):
    """The text encoder contract: it knows nothing of the implementation.

            The implementations: EmbeddingService (the live embedding slot) and the
            test HashEmbedder (tests/fakes.py) - the consumers (SearchService,
            DeduplicationService, the worker) depend on the protocol, not the client
            class.

    """

    def embed(self, text: str) -> list[float]:
        """Encode one text (the synchronous save/update/query path)."""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts; the result order = the input order."""
        ...

    def probe(self) -> str:
        """A cheap probe knock of the embedding slot (No39, arch §6.4): `check()`
                    without generation. The outcome is served as is (CheckResult).
        """
        ...


class EmbeddingService:
    """Encode texts through the embedding slot's embed (batch).

            It also maintains `last_attempt_ok` (NFR-4) - the source of
            `/health.embedding_ok`: None - no attempts yet; True/False - the last
            attempt's outcome. The status updates in `embed_texts` - the single point
            of all encodings (save/update/query/the background worker).

    """

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport
        | Callable[[httpx.Request], httpx.Response]
        | None = None,
        *,
        llm: LLMClient | None = None,
    ) -> None:
        self._settings = settings
        # A DI build (build_services passes the slot's shared client) or an own
        # client with a transport injection for unit tests (MockTransport).
        # The `embed` operation (LSBN-F-34): the flat EMBED_* env overrides the
        # encoding model/provider/address/key/timeout over the embedding slot
        # (which has no timeout of its own - the 720 s default).
        self._llm = (
            llm
            if llm is not None
            else LLMClient(
                SlotSpec.for_operation("embed", settings)[0], transport=transport
            )
        )
        # None - no attempts yet (health does not lie before the first data).
        self.last_attempt_ok: bool | None = None

    def embed(self, text: str) -> list[float]:
        """Encode one text (the synchronous save/update/query path)."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts; the result order = the input order."""
        if not texts:
            raise ValueError("embed_texts: пустой список текстов")
        try:
            embeddings = self._request_and_parse(list(texts))
        except EmbeddingError:
            self.last_attempt_ok = False  # the degradation will show in /health
            raise
        self.last_attempt_ok = True
        return embeddings

    def _request_and_parse(self, texts: list[str]) -> list[list[float]]:
        """The loop with the single retry + the response contract checks.

                    The retry - by the client's `LLMError.transient` signal
                    (429/5xx/transport/timeout); 401/403 and other 4xx never retry. The
                    client itself never retries - the repeat decision belongs to the
                    caller.

        """
        expected = len(texts)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            final = attempt == MAX_ATTEMPTS
            try:
                vectors = self._llm.embed(texts)
            except LLMError as exc:
                if final or not exc.transient:
                    # The call outcome (No39, arch §6.4): one type at the service boundary.
                    raise EmbeddingError(
                        str(exc), outcome=classify_llm_error(exc)
                    ) from exc
                continue  # the single retry (ARCH §3.2)
            return self._check_contract(vectors, expected)
        raise EmbeddingError("неожиданный выход из цикла попыток")  # unreachable

    def close(self) -> None:
        """Close the HTTP pool (a clean process shutdown)."""
        self._llm.close()

    def probe(self) -> str:
        """A cheap probe knock of the embedding slot (No39, arch §6.4): `check()`
                    without generation. The outcome is served as is (CheckResult).
        """
        return self._llm.check()

    #--- internal ---------------------------------------------------------

    def _check_contract(
        self, embeddings: object, expected: int
    ) -> list[list[float]]:
        """Check the embed response contract; any violation - EmbeddingError.

                    The vector count must equal the input count and every vector's
                    dimension - EMBEDDING_DIM (the vec0 table pins the dimension - a
                    mismatch would otherwise break the restart).

        """
        if not isinstance(embeddings, list) or len(embeddings) != expected:
            raise EmbeddingError(
                f"embed вернул {len(embeddings) if isinstance(embeddings, list) else 'не-список'} "
                f"векторов на {expected} вход(ов)"
            )
        dim = self._settings.embedding_dim
        for vector in embeddings:
            if not isinstance(vector, list) or len(vector) != dim:
                raise EmbeddingError(
                    f"размерность вектора не совпадает с EMBEDDING_DIM: ожидалась {dim}"
                )
        return embeddings