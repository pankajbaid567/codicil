#!/bin/bash
# Weekly: re-fetch the RBI pages, and if documents changed, rebuild the index (changed chunks only), upload it to S3 Vectors,
# restart the service on it, and roll back to the previous index if the new one does not come up healthy.
set -uo pipefail
DATA=/opt/codicil-data
IMG=codicil:latest
mkdir -p "$DATA"
exec >>/var/log/codicil-refresh.log 2>&1
echo "=== $(date -u +%FT%TZ) corpus refresh"

previous=""
if [ -e "$DATA/index/current" ]; then previous=$(readlink "$DATA/index/current"); fi

docker run --rm --user root -v "$DATA":/data "$IMG" \
  python scripts/refresh_corpus.py --data-dir /data --index-root /data/index --apply --max-changed 4
code=$?
case $code in
  0) echo "nothing to apply"; exit 0 ;;
  2) echo "some pages could not be checked; nothing applied"; exit 0 ;;
  10|11|12) echo "not applied (exit $code): needs a person to look; see the output above"; exit "$code" ;;
  20) ;;
  *) echo "refresh failed (exit $code)"; exit "$code" ;;
esac

echo "applied: uploading the new index to S3 Vectors"
# host network: the instance metadata service (IMDSv2, hop limit 1) is only reachable from the host network, which is where the role credentials come from
if ! docker run --rm --network host --user root -v "$DATA":/data -e AWS_DEFAULT_REGION=ap-south-1 \
      -e CODICIL_INDEX_DIR=/data/index/current -e CODICIL_S3VECTORS_BUCKET=codicil-vectors -e CODICIL_S3VECTORS_REGION=ap-south-1 \
      "$IMG" python scripts/upload_s3vectors.py; then
  echo "S3 Vectors upload failed; the service will fall back to the in-process index for that backend" >&2
fi

if /opt/codicil-src/deploy/aws-ec2/codicil-run.sh; then
  echo "service restarted on the refreshed index"
  exit 0
fi
echo "new index did not come up healthy; rolling back" >&2
if [ -n "$previous" ]; then ln -sfn "$previous" "$DATA/index/current"; else rm -f "$DATA/index/current"; fi
/opt/codicil-src/deploy/aws-ec2/codicil-run.sh || true
exit 30
