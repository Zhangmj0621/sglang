#!/bin/bash
#
# Overlay HiCache L3 PP-consistency sources onto an sglang install inside a
# container, without a source rebuild.
#
# Why this exists: the sglang trees baked into deployment images can lag this
# repo by hundreds of commits. Copying a whole newer checkout over such a tree
# breaks at import time, because newer modules reference symbols the older base
# does not define. The HiCache L3 changes, however, are confined to a handful of
# mem_cache / cache-controller modules whose imports still resolve against an
# older base -- so those files can be dropped in as-is, and the one change that
# lands outside that set (a two-line removal in scheduler.py) is applied to the
# target's own copy instead of overwriting it.
#
# The target install must be an editable/source checkout (pip install -e), so
# replacing .py files takes effect on the next process start.
#
# Usage:
#   scripts/overlay_hicache_pp.sh --src <repo>/python --dst <install>/python [--check-only]
#
#   --src         python/ dir of this repo (source of the overlay files)
#   --dst         python/ dir of the target install, e.g.
#                 /sgl-workspace/sglang/python
#   --check-only  verify only: report what would change, touch nothing
#
# Typically run inside the container:
#   docker cp <repo>/python <container>:/tmp/overlay-src
#   docker exec <container> bash scripts/overlay_hicache_pp.sh \
#       --src /tmp/overlay-src --dst /sgl-workspace/sglang/python
#
# Originals are saved next to the install as sglang.hicache-pp-backup/, and a
# re-run restores from that backup first, so repeated runs converge on the same
# result. Rollback is a single cp -rp from the backup.

set -euo pipefail

SRC=""
DST=""
CHECK_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --src) SRC="${2:?--src needs a path}"; shift 2 ;;
    --dst) DST="${2:?--dst needs a path}"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$SRC" ] && [ -n "$DST" ] || { echo "usage: $0 --src <repo>/python --dst <install>/python [--check-only]" >&2; exit 2; }
[ -d "$SRC/sglang" ] || { echo "FATAL: --src is not a python/ dir with sglang/: $SRC" >&2; exit 1; }
[ -d "$DST/sglang" ] || { echo "FATAL: --dst is not a python/ dir with sglang/: $DST" >&2; exit 1; }

# Keep the backup outside the target checkout so it never shows up in its
# git status: --dst is <root>/python, so go up twice to sit beside <root>.
BACKUP="$(cd "$DST/../.." && pwd)/$(basename "$(cd "$DST/.." && pwd)").hicache-pp-backup"

# Files copied verbatim. Their sglang imports resolve against an older base.
OVERLAY_FILES=(
  "sglang/srt/managers/cache_controller.py"
  "sglang/srt/mem_cache/hicache_storage.py"
  "sglang/srt/mem_cache/hiradix_cache.py"
  "sglang/srt/mem_cache/unified_radix_cache.py"
  "sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py"
  "sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py"
)

# Patched in place: this repo's scheduler.py may pull in symbols an older base
# lacks, so only the HiCache-relevant hunk is applied.
SCHED="sglang/srt/managers/scheduler.py"

echo "=== 1/5 checking preconditions ==="
for f in "${OVERLAY_FILES[@]}"; do
  [ -f "$SRC/$f" ] || { echo "FATAL: missing in --src: $f" >&2; exit 1; }
  [ -f "$DST/$f" ] || { echo "FATAL: missing in --dst: $f" >&2; exit 1; }
done
[ -f "$DST/$SCHED" ] || { echo "FATAL: missing in --dst: $SCHED" >&2; exit 1; }

changed=0
for f in "${OVERLAY_FILES[@]}"; do
  if cmp -s "$SRC/$f" "$DST/$f"; then
    echo "  same  $f"
  else
    echo "  DIFF  $f"
    changed=$((changed + 1))
  fi
done
echo "$changed of ${#OVERLAY_FILES[@]} overlay files differ"

if [ "$CHECK_ONLY" = 1 ]; then
  echo
  echo "--check-only: nothing was modified."
  exit 0
fi

echo "=== 2/5 backup / restore ==="
if [ -d "$BACKUP" ]; then
  echo "  backup exists -> restoring originals first"
  for f in "${OVERLAY_FILES[@]}" "$SCHED"; do
    [ -f "$BACKUP/$f" ] && cp -p "$BACKUP/$f" "$DST/$f"
  done
else
  for f in "${OVERLAY_FILES[@]}" "$SCHED"; do
    mkdir -p "$BACKUP/$(dirname "$f")"
    cp -p "$DST/$f" "$BACKUP/$f"
  done
  echo "  saved originals -> $BACKUP"
fi

echo "=== 3/5 overlaying ${#OVERLAY_FILES[@]} files ==="
for f in "${OVERLAY_FILES[@]}"; do
  cp -p "$SRC/$f" "$DST/$f"
  echo "  overlaid  $f"
done

echo "=== 4/5 patching $SCHED ==="
python - "$DST/$SCHED" <<'PY'
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    src = fh.read()

# The unified cache no longer exposes terminate_prefetch(); aborts go through
# release_aborted_request() for every hierarchical-cache configuration.
OLD = """                if self.enable_hicache_storage:
                    # Release prefetch events associated with the request
                    self.tree_cache.release_aborted_request(candidate_req.rid)
                elif self.enable_hierarchical_cache:
                    self.tree_cache.terminate_prefetch(candidate_req.rid)
"""
NEW = """                if self.enable_hicache_storage:
                    # Release prefetch events associated with the request
                    self.tree_cache.release_aborted_request(candidate_req.rid)
"""

if OLD not in src and NEW in src:
    print("  already patched")
elif src.count(OLD) != 1:
    print(f"  FATAL: expected 1 match of the abort branch, found {src.count(OLD)}")
    sys.exit(1)
else:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src.replace(OLD, NEW, 1))
    print("  removed the terminate_prefetch abort branch")
PY

echo "=== 5/5 verifying ==="
for f in "${OVERLAY_FILES[@]}" "$SCHED"; do
  python -c "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read())" "$DST/$f"
  echo "  parsed OK  $f"
done

if grep -rqs "tree_cache.terminate_prefetch" "$DST/sglang/srt/"; then
  echo "  WARNING: tree_cache.terminate_prefetch callers remain:"
  grep -rns "tree_cache.terminate_prefetch" "$DST/sglang/srt/" || true
else
  echo "  no tree_cache.terminate_prefetch callers remain"
fi

python - <<'PY'
import importlib
import sys
import traceback

MODULES = [
    "sglang.srt.mem_cache.hicache_storage",
    "sglang.srt.managers.cache_controller",
    "sglang.srt.mem_cache.hiradix_cache",
    "sglang.srt.mem_cache.unified_radix_cache",
    "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller",
    "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store",
    "sglang.srt.managers.scheduler",
]

failures = 0
for name in MODULES:
    try:
        importlib.import_module(name)
        print(f"  import OK  {name}")
    except Exception as exc:  # noqa: BLE001 - report and keep going
        failures += 1
        print(f"  IMPORT FAIL {name} :: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=6)

if not failures:
    from sglang.srt.managers.cache_controller import HiCacheController, PrefetchAck
    from sglang.srt.mem_cache.hicache_storage import count_pool_hits
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    expectations = [
        ("PrefetchAck", PrefetchAck is not None),
        ("count_pool_hits", callable(count_pool_hits)),
        ("prefetch_sync_thread_func", hasattr(HiCacheController, "prefetch_sync_thread_func")),
        ("_page_transfer_kv_batch", hasattr(HiCacheController, "_page_transfer_kv_batch")),
        ("_create_sync_groups", hasattr(HiCacheController, "_create_sync_groups")),
        ("PrefetchOperation.increment removed", not hasattr(HiCacheController, "increment")),
        ("_handle_prefetch_result", hasattr(UnifiedRadixCache, "_handle_prefetch_result")),
        ("_check_hybrid_prefetch_result", hasattr(UnifiedRadixCache, "_check_hybrid_prefetch_result")),
        ("UnifiedRadixCache.terminate_prefetch removed", not hasattr(UnifiedRadixCache, "terminate_prefetch")),
    ]
    for label, ok in expectations:
        print(("  symbol OK  " if ok else "  SYMBOL FAIL ") + label)
        failures += 0 if ok else 1

print("RESULT: " + ("PASS" if failures == 0 else f"FAIL ({failures})"))
sys.exit(1 if failures else 0)
PY

echo
echo "Done. Rollback with: cp -rp $BACKUP/sglang $DST/"
