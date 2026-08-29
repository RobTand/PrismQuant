#!/usr/bin/env bash
# Ambient box vitals for Claude's context — the agent equivalent of keeping
# btop open in another terminal (Rob, 2026-08-28: "I honestly just want live
# cpu/gpu performance and box metrics to be part of your context").
#
# Emits ONE compact line as PreToolUse additionalContext so every Bash call
# comes back stamped with current headroom, whether or not Claude thought to
# look. The failure this prevents: launching a large job into a box that has
# no room, then being surprised by the OOM.
#
# GB10 note: nvidia-smi reports [N/A] for memory.used/total on unified memory.
# /proc/meminfo MemAvailable IS the GPU memory ceiling here; do not substitute
# an nvidia-smi memory field, it does not exist on this hardware.
#
# Budget: must stay under ~50 ms. /proc reads are ~7 ms, one nvidia-smi call
# ~27 ms. Never ssh from this path — a remote host's numbers come from its own
# recorder via a cached file, or are omitted.
set -u
LC_ALL=C

# Peer box (lina). Its own netdata answers over the LAN in ~2 ms; one curl
# reuses a single connection for all four queries. Started FIRST and collected
# LAST so the round trip overlaps the local /proc work instead of adding to it.
# Never ssh from this path.
_peer_host=192.168.1.110
_pq() { printf 'http://%s:19999/api/v2/data?contexts=%s&after=-1&points=1&format=json2&group_by=context' "$_peer_host" "$1"; }
exec 3< <(curl -s -m 1 \
  "$(_pq mem.available)" "$(_pq system.load)" "$(_pq nvidia_smi.gpu_utilization)" \
  "http://$_peer_host:19999/api/v1/alarm_count" 2>/dev/null | jq -s -r '
    def v(i): (.[i].result.data[0][1]? // null) | if type=="array" then .[0] else . end;
    (v(0)) as $m | (v(1)) as $l | (v(2)) as $g | (.[3][0] // 0) as $a
    | if $m == null then "" else
        " | lina: mem \($m/1024|floor)G avail, load \($l|.*100|round/100), gpu \($g|round)%\(if $a>0 then ", \($a) alarm" else "" end)"
      end' 2>/dev/null)

read -r _ memtotal _ < <(grep -m1 '^MemTotal:' /proc/meminfo)
read -r _ memavail _ < <(grep -m1 '^MemAvailable:' /proc/meminfo)
read -r _ anon _ < <(grep -m1 '^AnonPages:' /proc/meminfo)
read -r _ shmem _ < <(grep -m1 '^Shmem:' /proc/meminfo)
read -r _ swapfree _ < <(grep -m1 '^SwapFree:' /proc/meminfo)

gib() { awk -v k="$1" 'BEGIN{printf "%.1f", k/1048576}'; }
avail_g=$(gib "$memavail"); total_g=$(gib "$memtotal")
anon_g=$(gib "$anon"); shmem_g=$(gib "$shmem")
avail_pct=$(awk -v a="$memavail" -v t="$memtotal" 'BEGIN{printf "%.0f", 100*a/t}')

gpu=$(timeout 2 nvidia-smi \
  --query-gpu=utilization.gpu,power.draw,temperature.gpu,clocks_event_reasons.active \
  --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$gpu" ]; then
  IFS=',' read -r g_util g_pow g_temp g_thr <<<"$gpu"
  # Power-vs-envelope leads, utilization trails. On GB10 utilization.gpu is
  # non-diagnostic under load -- it means "a kernel is resident", not "the SMs
  # are working", and reads 96% for a memory-stalled kernel exactly as for a
  # saturated one (measured 08-28: 96% on both sides of a 5.83x throughput
  # change). Power against the ~140 W envelope is the signal that moved, and
  # the envelope fraction doubles as a remaining-headroom estimate.
  # AGENTS.md principle 13 / CLAUDE.md principle 15.
  g_env=$(awk -v p="${g_pow// /}" 'BEGIN{if(p+0>0) printf "%d", (p/140)*100; else print ""}')
  if [ -n "$g_env" ]; then
    gpu_s="gpu ${g_pow// /}W/140W (${g_env}% envelope) util:${g_util// /}% ${g_temp// /}C"
  else
    gpu_s="gpu ${g_util// /}% ${g_pow// /}W ${g_temp// /}C"
  fi
  # 0x1 = GpuIdle is normal; anything else at load is worth seeing.
  case "${g_thr// /}" in 0x0000000000000000|0x0000000000000001) ;; *) gpu_s="$gpu_s thr:${g_thr// /}" ;; esac
else
  gpu_s="gpu n/a"
fi

read -r load1 _ < /proc/loadavg
ncpu=$(nproc 2>/dev/null || echo 1)

disk=$(df -BG --output=avail,pcent /home/rob 2>/dev/null | tail -1)
disk_avail=$(awk '{print $1}' <<<"$disk"); disk_used=$(awk '{print $2}' <<<"$disk")

# Largest resident process. The full ps sort costs ~40 ms, so only pay it
# when headroom is actually tight — that is the only time the answer matters.
top_s=""
if [ "${memavail}" -lt 31457280 ]; then
  top_proc=$(ps -eo rss=,comm= --sort=-rss 2>/dev/null | head -1)
  top_s=$(awk '{printf " | top %s %.1fG", $2, $1/1048576}' <<<"$top_proc")
fi

peer=""
IFS= read -r -t 1 peer <&3 2>/dev/null || peer=""
exec 3<&-

# Local netdata's verdict: active alarms across all ~2100 charts. The numbers
# above stay on /proc — on GB10 that is the only honest memory source, and
# netdata's own GPU framebuffer chart reports null there. jq not python: 12 vs 130 ms.
nd=$(curl -s -m 1 'http://127.0.0.1:19999/api/v1/alarms?active' 2>/dev/null | jq -r '
  [.alarms[]?] as $a
  | ($a | map(select(.status == "CRITICAL")) | length) as $c
  | if ($a|length) == 0 then ""
    else " | netdata \($c) CRIT/\(($a|length) - $c) warn: \($a | map(.name) | unique | .[0:3] | join(","))"
    end' 2>/dev/null)

warn=""
avail_int=${avail_g%.*}
if [ "$avail_int" -lt 6 ]; then warn="!! CRITICAL MEM "
elif [ "$avail_int" -lt 16 ]; then warn="! LOW MEM "
fi
[ "${swapfree}" = "0" ] && swap_s=" noswap" || swap_s=""

line=$(printf '%s[vitals sparky] mem %sG/%sG avail (%s%%, anon %sG, shm %sG%s) | %s | load %s/%s | disk %s free (%s used)%s%s%s' \
  "$warn" "$avail_g" "$total_g" "$avail_pct" "$anon_g" "$shmem_g" "$swap_s" \
  "$gpu_s" "$load1" "$ncpu" "$disk_avail" "$disk_used" "$top_s" "$peer" "$nd")

# --hook <event>: emit the hook JSON envelope so the line lands in the model's
# context rather than only in the transcript. Plain stdout is the default so
# the script stays runnable by hand.
if [ "${1:-}" = "--hook" ] && command -v jq >/dev/null 2>&1; then
  jq -cn --arg e "${2:-PreToolUse}" --arg c "$line" \
    '{hookSpecificOutput:{hookEventName:$e,additionalContext:$c},suppressOutput:true}'
else
  printf '%s\n' "$line"
fi
