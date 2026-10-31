#!/usr/bin/env bash
# Run a command and kill it before the box starts swapping.
#
#   tools/memguard.sh [--min-gib N] -- command args...
#
# Polls MemAvailable twice a second; when it drops under N GiB (default 3) the
# command's whole process group gets SIGKILL, and the exit status says so (137).
# The user cgroup here delegates no memory controller, so `systemd-run -p
# MemoryMax` is silently ignored, and `ulimit -v` breaks CUDA's address-space
# reservations. A watchdog is what is left. A Diffusers run with cpu offload
# froze this 32 GB machine on 2026-10-12; this is the fence against a repeat.

set -u
min_gib=3
if [ "${1:-}" = "--min-gib" ]; then min_gib=$2; shift 2; fi
[ "${1:-}" = "--" ] && shift

min_kib=$((min_gib * 1048576))
setsid "$@" &
pid=$!
pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
killed=0
while kill -0 "$pid" 2>/dev/null; do
    avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    if [ "$avail" -lt "$min_kib" ]; then
        echo "memguard: MemAvailable $((avail / 1024)) MiB < $min_gib GiB, killing pgid $pgid" >&2
        kill -KILL -- "-$pgid" 2>/dev/null
        killed=1
        break
    fi
    sleep 0.5
done
wait "$pid" 2>/dev/null
status=$?
[ "$killed" = 1 ] && exit 137
exit $status
