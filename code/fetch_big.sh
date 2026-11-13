#!/usr/bin/env bash
# Resilient (auto-resume) download of one large file from the HF mirror.
# usage: fetch_big.sh <url> <outfile> <expected_bytes>
set -u
URL="$1"; OUT="$2"; EXPECT="$3"
for i in $(seq 1 100); do
  CUR=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
  if [ "$CUR" -ge "$EXPECT" ]; then echo "complete: $CUR bytes"; exit 0; fi
  echo "[try $i] have $CUR / $EXPECT"
  curl -sS --ssl-no-revoke -L -C - --max-time 1800 --speed-time 120 --speed-limit 1024 -o "$OUT" "$URL"
  RC=$?
  echo "[try $i] curl rc=$RC"
  if [ $RC -eq 33 ]; then echo "server rejected range, restarting"; rm -f "$OUT"; fi
  sleep 4
done
echo "gave up"; exit 1
