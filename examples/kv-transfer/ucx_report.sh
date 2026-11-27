#!/bin/bash
# Which UCX can this image use: the apt one (a tool only) and the one the nixl
# wheel loads. Run at build time (no IB device: modules and links only) and by
# rdma_smoke.yaml on a node (devices too). Never fails.
echo "== system UCX"
ucx_info -v 2>&1 | head -8
ucx_info -d 2>&1 | grep -E "Transport|Device" | sort | uniq -c

echo "== UCX in the nixl wheel"
site=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
libs=$(find "$site" \( -name 'libucp*' -o -name 'libucs*' -o -name 'libuct*' \) 2>/dev/null | sort)
if [ -z "$libs" ]; then
  echo "no libucp/libuct under $site: the wheel does not bundle UCX, or keeps it elsewhere"
  pip show -f nixl 2>/dev/null | grep -i -E "^(version|requires)"
  exit 0
fi
echo "$libs"
dir=$(dirname "$(echo "$libs" | grep libucp | head -1)")
echo "-- IB modules (libuct_ib*, libuct_rdmacm*) and what they fail to link"
for f in $(echo "$libs" | grep -E 'libuct_(ib|rdmacm)'); do
  echo "$f"; ldd "$f" 2>&1 | grep -E "not found" || echo "  links ok"
done
echo "-- ucx_info (the system binary run on the wheel's libraries: approximate)"
LD_LIBRARY_PATH="$dir:$LD_LIBRARY_PATH" UCX_MODULE_DIR="$dir/ucx" \
  ucx_info -d 2>&1 | grep -E "Transport|Device" | sort | uniq -c
exit 0
