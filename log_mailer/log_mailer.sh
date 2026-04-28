#!/bin/bash
# Email the last N lines of a training log on a fixed interval.
# Sends an immediate first email, then repeats every INTERVAL seconds.
#
# Usage:
#   nohup bash log_mailer.sh >> ~/log_mailer.log 2>&1 &
#
# Override: LOGFILE, EMAIL, INTERVAL (seconds), TAIL_LINES.

LOGFILE="${LOGFILE:-/home/export/sdurgut/scratch/ttHbb_SPANet/logs/classifier/btag_TML_QvG/classifier_btag_TML_QvG_1006.out}"
EMAIL="${EMAIL:-sdurgut@andrew.cmu.edu}"
INTERVAL="${INTERVAL:-3600}"
TAIL_LINES="${TAIL_LINES:-10}"

echo "[log_mailer] started $(date -Is)  LOGFILE=$LOGFILE  EMAIL=$EMAIL  INTERVAL=${INTERVAL}s  TAIL_LINES=$TAIL_LINES"

send_snapshot() {
  ts="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  subj="Falcon log snapshot — $ts"

  if [[ -f "$LOGFILE" ]]; then
    tail -n "$TAIL_LINES" "$LOGFILE" | mailx -s "$subj" "$EMAIL" \
      || echo "[log_mailer] mailx failed $(date -Is)" >&2
  else
    printf '%s\n' "Log file not found (yet): $LOGFILE" \
      | mailx -s "Falcon log mailer — MISSING — $ts" "$EMAIL" || true
  fi
}

send_snapshot

while true; do
  sleep "$INTERVAL"
  send_snapshot
done
