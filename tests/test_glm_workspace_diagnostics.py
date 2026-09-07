"""CPU-only checks for the bounded workspace experiment's failure evidence."""
import json
import sys
import weakref

import pytest
import torch

from experiments.glm_layer_workspace import (
    audit_collector_rows, failed_collector_row_counts, finalize_workspace, watch_memory_samples,
)


def test_failure_rows_are_copied_without_tensor_or_traceback_ownership():
    refs = []
    counts = {'observed': 19, 'unobserved': 0}

    def collector():
        seen = counts
        hessian = torch.ones(2, 2)
        refs.append(weakref.ref(hessian))
        raise RuntimeError('after some callbacks')

    def run():
        try:
            collector()
        except RuntimeError as exc:
            return failed_collector_row_counts(exc, collector, counts)

    record = run()
    assert record['status'] == 'observed_before_failure'
    assert record['scope'] == 'partial_failed_collection_not_census'
    assert record['rows'] == {'observed': 19, 'unobserved': 0}
    assert record['positive_row_qnames'] == ['observed']
    assert record['zero_row_qnames'] == ['unobserved']
    counts['observed'] = 99
    assert record['rows']['observed'] == 19
    assert refs[0]() is None
    json.dumps(record)


@pytest.mark.parametrize('kind', ['absent_frame', 'missing_key', 'negative', 'noninteger'])
def test_unknown_failure_counts_never_become_zero_coverage(kind):
    def collector():
        seen = {'unit': 1}
        if kind == 'missing_key':
            seen = {}
        if kind == 'negative':
            seen['unit'] = -1
        if kind == 'noninteger':
            seen['unit'] = torch.tensor(1)
        raise RuntimeError('failure')

    def other():
        seen = {'unit': 0}
        raise RuntimeError('different frame')

    # Even the same function name is insufficient; match the actual code object.
    other.__name__ = collector.__name__
    try:
        (other if kind == 'absent_frame' else collector)()
    except RuntimeError as exc:
        record = failed_collector_row_counts(exc, collector, ['unit'])
    assert record['status'] == 'unknown'
    assert record['expected_qnames'] == ['unit']
    assert 'rows' not in record and 'zero_row_qnames' not in record


def test_memory_observation_continues_after_guard_and_transient_read_failure():
    class Stop:
        count = 0

        def is_set(self):
            return self.count == 4

        def wait(self, interval):
            assert interval == 0.1
            self.count += 1

    stopped = Stop()
    records, errors = [], []

    def observe():
        if stopped.count == 1:
            raise OSError('temporary cgroup read failure')
        return dict(sample=stopped.count, guard_tripped=True)

    watch_memory_samples(stopped, observe, records.append, lambda exc: errors.append(str(exc)))
    assert [row['sample'] for row in records] == [0, 2, 3]
    assert all(row['guard_tripped'] for row in records)
    assert errors == ['temporary cgroup read failure']


@pytest.mark.parametrize('collection_fails', [False, True])
def test_shutdown_failure_still_finalizes_evidence_and_preserves_collection_error(collection_fails):
    result = {'status': 'running'}
    finalized = []
    original = ValueError('collector failed')

    def shutdown():
        raise RuntimeError('prefetch shutdown failed')

    def run():
        try:
            if collection_fails:
                raise original
        finally:
            finalize_workspace(shutdown, lambda: finalized.append(dict(result)),
                               result, sys.exc_info()[1])

    with pytest.raises(ValueError if collection_fails else RuntimeError) as caught:
        run()
    if collection_fails:
        assert caught.value is original
    else:
        assert str(caught.value) == 'prefetch shutdown failed'
    assert result['status'] == 'failed'
    assert result['cleanup_failure'] == dict(type='RuntimeError', message='prefetch shutdown failed')
    assert finalized == [result]


@pytest.mark.parametrize('guard_fails', [False, True])
def test_returned_counts_survive_later_guard_or_zero_row_failure(guard_fails):
    record = {}
    counts = {'positive': 12, 'zero': 0}
    checks = []

    def check_guard(label):
        checks.append(label)
        if guard_fails:
            raise RuntimeError('post-collection memory guard')

    with pytest.raises(RuntimeError, match='memory guard' if guard_fails else 'no observed rows'):
        audit_collector_rows(counts, list(counts), record, check_guard)
    counts['positive'] = 50
    assert checks == ['after_collection']
    assert record['returned_target_row_observations']['rows'] == {'positive': 12, 'zero': 0}
    assert record['returned_target_row_observations']['zero_row_qnames'] == ['zero']
    assert record['returned_target_row_observations']['status'] == 'collector_returned'
