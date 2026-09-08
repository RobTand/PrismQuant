"""Bounded host evidence selection; HTTP and clocks are controlled on CPU."""
import io
import json
from urllib.parse import parse_qs, urlparse

import pytest

from experiments.workspace_netdata import NetdataWriter, sample_netdata


BASE_CHARTS = (
    'system.cpu', 'system.load', 'system.ram', 'system.io',
    'system.cpu_some_pressure', 'system.memory_some_pressure', 'system.io_some_pressure',
    'mem.available', 'mem.reclaiming', 'mem.thp_details',
    'nfs.rpc', 'nfs.proc4', 'prismabuild.mount_latency', 'prismabuild.mount_probe_state',
)
GPU_SUFFIXES = ('power_draw', 'clock_freq', 'frame_buffer_memory_usage',
                'pcie_bandwidth_utilization')


def payload(gpu='new-unlisted-GPU-123'):
    names = list(BASE_CHARTS) + [f'nvidia_smi.gpu_{gpu}_{suffix}' for suffix in GPU_SUFFIXES]
    return {name: dict(name=name, context=name, units='unit', last_updated=123,
                      dimensions={'value': {'name': 'redundant', 'value': 7}}) for name in names}


def opener_for(metrics, seen=None):
    def opener(url, timeout):
        if seen is not None:
            seen.append(url)
        assert timeout == 5
        return io.BytesIO(json.dumps(metrics).encode())
    return opener


def test_server_filter_retains_required_charts_and_unknown_gpu_uuid():
    raw = payload()
    raw['systemd.unrelated_unit'] = raw['system.cpu']
    urls = []
    record = sample_netdata('sparky', opener_for(raw, urls))
    query = parse_qs(urlparse(urls[0]).query)
    assert query['format'] == ['json']
    assert 'nvidia_smi.*_power_draw' in query['filter'][0]
    assert 'new-unlisted-GPU-123' not in query['filter'][0]
    assert record['schema'] == 'prismaquant.workspace_netdata.v2'
    assert record['scope'] == 'cpu_memory_nfs_mount_gpu_power_clocks_memory_pcie'
    assert set(record['metrics']) == set(payload())
    assert record['metrics']['system.cpu']['dimensions']['value'] == {'value': 7}
    assert record['response_bytes'] > 0


@pytest.mark.parametrize('missing', ['system.cpu', 'system.memory_some_pressure', 'nfs.rpc',
                                     'prismabuild.mount_latency',
                                     'nvidia_smi.gpu_new-unlisted-GPU-123_power_draw',
                                     'nvidia_smi.gpu_new-unlisted-GPU-123_clock_freq'])
def test_missing_required_chart_refuses_evidence(missing):
    raw = payload()
    del raw[missing]
    with pytest.raises(RuntimeError, match='required'):
        sample_netdata('sparklina', opener_for(raw))


def test_response_cap_refuses_before_unbounded_json_parse():
    raw = payload()
    raw['huge'] = 'x' * (256 * 1024)
    with pytest.raises(RuntimeError, match='response.*cap'):
        sample_netdata('sparky', opener_for(raw))


def test_http_failure_is_not_a_valid_sample():
    def unavailable(url, timeout):
        raise OSError('API unavailable')
    with pytest.raises(OSError, match='API unavailable'):
        sample_netdata('sparky', unavailable)


def test_per_record_and_total_caps_preserve_complete_prior_lines():
    out = io.StringIO()
    record = sample_netdata('sparky', opener_for(payload()))
    size = len((json.dumps(record, separators=(',', ':'))+'\n').encode())
    writer = NetdataWriter(out, max_bytes=size)
    writer.write(record)
    saved = out.getvalue()
    with pytest.raises(RuntimeError, match='total.*cap'):
        writer.write(record)
    assert out.getvalue() == saved
    assert json.loads(saved) == record
    assert writer.bytes_written == size
    with pytest.raises(RuntimeError, match='record.*cap'):
        NetdataWriter(io.StringIO()).write({'huge': 'x' * (64 * 1024)})
