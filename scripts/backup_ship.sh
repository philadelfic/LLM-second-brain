#!/bin/sh
# backup_ship.sh — template of a shipping command for BACKUP_SHIP_CMD (LSB, №14).
#
# The service runs this command once per snapshot, right after the snapshot has
# been VERIFIED. It passes the file to ship in the environment:
#
#   LSB_BACKUP_FILE — absolute path of the snapshot inside the container
#   LSB_BACKUP_NAME — its file name
#
# The destination and the credentials live OUTSIDE this repository and outside
# the image. Put a config file (rclone/rsync/ssh) and the keys on the HOST, then
# bind-mount them read-only into the container and reference that path from the
# command, e.g.:
#
#   # docker-compose.override.yml
#   environment:
#     BACKUP_SHIP_CMD: "/srv/ship/backup_ship.sh"
#     BACKUP_SHIP_TIMEOUT_SEC: "300"
#   volumes:
#     - /srv/ship:/srv/ship:ro
#
# Contract: on success the command MUST print `size=<bytes>` on stdout, and the
# service counts the copy as shipped only when the exit code is 0 AND that size
# equals the local snapshot size. The command and its output are NOT logged by
# the service (nothing secret may reach the logs) — keep the command quiet
# anyway. Ship copies only into a TRUSTED network: a snapshot is the whole
# memory store. Never hardcode a password/token here — read it from the mounted
# config or a key file.
set -eu

: "${LSB_BACKUP_FILE:?LSB_BACKUP_FILE is not set}"

# --- replace the two lines below with your own destination -------------------
# rclone:
#   rclone --config /srv/ship/rclone.conf copy "$LSB_BACKUP_FILE" remote:lsb-backups/
# rsync over ssh with a mounted key:
#   rsync -a -e "ssh -i /srv/ship/id_ed25519 -o UserKnownHostsFile=/srv/ship/known_hosts" \
#         "$LSB_BACKUP_FILE" backup@example-host:/srv/lsb-backups/
#
# The stub below only proves the contract locally: it reports the file size.
# -----------------------------------------------------------------------------
size=$(stat -c%s "$LSB_BACKUP_FILE")
echo "size=$size"
