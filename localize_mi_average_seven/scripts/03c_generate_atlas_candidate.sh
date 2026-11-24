#!/usr/bin/env bash
# Generate a separate watershed candidate; does not adopt or approve surfaces.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
subject="$repo/localize_mi_average_seven/work/freesurfer/average_seven"
candidate="$repo/localize_mi_average_seven/work/bem_candidates/atlas_mode"
test -f "$subject/mri/T1.mgz"
sudo -v
mkdir -p "$(dirname "$candidate")"
mkdir "$candidate"
sudo docker run --rm \
  --user "$(id -u):$(id -g)" \
  --volume "$repo:$repo:ro" \
  --volume "$candidate:$candidate:rw" \
  --volume /home/girvi949/.freesurfer/license.txt:/license.txt:ro \
  --env FS_LICENSE=/license.txt \
  --entrypoint mri_watershed freesurfer/freesurfer:8.2.0 \
  -T1 -atlas -useSRAS -surf "$candidate/average_seven" \
  "$subject/mri/T1.mgz" "$candidate/ws.mgz" \
  > "$candidate/watershed.log" 2>&1
echo "Candidate generated; nesting and MRI review are still required."
tail -n 20 "$candidate/watershed.log"
