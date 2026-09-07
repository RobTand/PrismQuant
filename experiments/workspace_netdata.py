"""Bounded host evidence for the workspace diagnostic, never model telemetry.

The existing allmetrics API supports server-side chart patterns:
https://learn.netdata.cloud/docs/exporting-metrics/exporting-reference#chart-filtering
"""
import fnmatch
import json
import time
import urllib.parse
import urllib.request

SCHEMA = 'prismaquant.workspace_netdata.v2'
SCOPE = 'cpu_memory_nfs_mount_gpu_power_clocks_memory_pcie'
CHART_PATTERNS = (
    'system.cpu', 'cpu.cpu*', 'system.load', 'system.ram', 'system.io',
    'system.*pressure*', 'mem.available', 'mem.swap*', 'mem.pgfaults',
    'mem.oom_kill', 'mem.kernel', 'mem.slab', 'mem.thp*', 'mem.reclaiming',
    'mem.writeback', 'nfs.*', 'prismabuild.mount_*',
    'nvidia_smi.*_power_draw', 'nvidia_smi.*_clock_freq',
    'nvidia_smi.*_temperature', 'nvidia_smi.*_frame_buffer_memory_usage',
    'nvidia_smi.*_bar1_memory_usage', 'nvidia_smi.*_pcie_bandwidth*',
)
REQUIRED_CHARTS = frozenset((
    'system.cpu', 'system.load', 'system.ram', 'system.io',
    'system.cpu_some_pressure', 'system.memory_some_pressure', 'system.io_some_pressure',
    'mem.available', 'mem.reclaiming', 'mem.thp_details',
    'nfs.rpc', 'nfs.proc4', 'prismabuild.mount_latency', 'prismabuild.mount_probe_state',
))
REQUIRED_GPU_SUFFIXES = ('clock_freq', 'frame_buffer_memory_usage',
                         'pcie_bandwidth_utilization')
MAX_RESPONSE_BYTES = 256 * 1024
MAX_RECORD_BYTES = 64 * 1024
MAX_TOTAL_BYTES = 256 * 1024**2
MAX_CHARTS = 128


def sample_netdata(host, opener=urllib.request.urlopen):
    query = urllib.parse.urlencode({'format': 'json', 'filter': ' '.join(CHART_PATTERNS)})
    with opener(f'http://{host}:19999/api/v1/allmetrics?{query}', timeout=5) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RuntimeError('Netdata response exceeded byte cap')
    metrics = json.loads(raw)
    selected = {name: value for name, value in metrics.items()
                if any(fnmatch.fnmatchcase(name, pattern) for pattern in CHART_PATTERNS)}
    missing = REQUIRED_CHARTS - selected.keys()
    gpu_stems = [name.removesuffix('_power_draw') for name in selected
                 if fnmatch.fnmatchcase(name, 'nvidia_smi.*_power_draw')]
    if not gpu_stems:
        raise RuntimeError('Netdata required GPU power chart missing')
    missing |= {f'{stem}_{suffix}' for stem in gpu_stems for suffix in REQUIRED_GPU_SUFFIXES
                if f'{stem}_{suffix}' not in selected}
    if missing:
        raise RuntimeError(f'Netdata required charts missing: {sorted(missing)}')
    if len(selected) > MAX_CHARTS:
        raise RuntimeError('Netdata selected chart count exceeded cap')
    compact = {}
    for name, chart in selected.items():
        compact[name] = {key: chart[key] for key in ('context', 'units', 'last_updated')}
        compact[name]['dimensions'] = {
            dimension: {'value': value['value']}
            for dimension, value in chart['dimensions'].items()}
    return dict(schema=SCHEMA, scope=SCOPE, host=host, time=time.time(),
                response_bytes=len(raw), metrics=compact)


class NetdataWriter:
    """Write whole samples only; refuse limits instead of silently truncating."""
    def __init__(self, out, max_bytes=MAX_TOTAL_BYTES):
        self.out = out
        self.bytes_written = 0
        self.max_bytes = max_bytes

    def write(self, record):
        data = json.dumps(record, separators=(',', ':'))+'\n'
        size = len(data.encode())
        if size > MAX_RECORD_BYTES:
            raise RuntimeError('Netdata record exceeded byte cap')
        if self.bytes_written + size > self.max_bytes:
            raise RuntimeError('Netdata total output exceeded byte cap')
        self.out.write(data)
        self.out.flush()
        self.bytes_written += size
