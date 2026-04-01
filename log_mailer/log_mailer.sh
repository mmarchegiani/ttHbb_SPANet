#!/bin/bash
# Email the full training log on a fixed interval (default: 30 minutes).
# Run in the background, e.g.:
#   nohup env LOGFILE=/path/to/job.out ./log_mailer.sh >> ~/log_mailer.log 2>&1 &
#
# Override: LOGFILE, EMAIL, INTERVAL (seconds).

LOGFILE="${LOGFILE:-/home/export/sdurgut/scratch/ttHbb_SPANet/logs/classifier/btag_T/classifier_btag_T_529.out}"
EMAIL="${EMAIL:-sdurgut@andrew.cmu.edu}"
INTERVAL="${INTERVAL:-1800}"

echo "[log_mailer] started $(date -Is)  LOGFILE=$LOGF`ILE  EMAIL=$EMAIL  INTERVAL=${INTERVAL}s"

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  subj="Falcon log snapshot — $ts"

  if [[ -f "$LOGFILE" ]]; then
    mailx -s "$subj" "$EMAIL" < "$LOGFILE" || echo "[log_mailer] mailx failed $(date -Is)" >&2
  else
    printf '%s\n' "Log file not found (yet): $LOGFILE" | mailx -s "Falcon log mailer — MISSING — $ts" "$EMAIL" || true
  fi

  sleep "$INTERVAL"
done
