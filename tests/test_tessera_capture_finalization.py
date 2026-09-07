"""The collector must release obsolete capture buffers during finalization."""
import weakref
import torch
from prismaquant import tessera_campaign as tc


def test_scoring_chunks_are_released_between_units(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(4,4,bias=False),
                                torch.nn.Linear(4,4,bias=False)).to(torch.bfloat16)
    tokens = [torch.arange(12).reshape(3,4).to(torch.bfloat16)]
    original_cat = torch.cat
    previous = []
    def concatenate(chunks, *args, **kwargs):
        assert all(ref() is None for ref in previous), 'previous unit scoring chunks retained'
        previous[:] = [weakref.ref(value) for value in chunks]
        return original_cat(chunks, *args, **kwargs)
    monkeypatch.setattr(torch, 'cat', concatenate)
    rows, hessians, counts, maxima = tc._collect_activations(
        model,['0','1'],tokens,2,'cpu',want_hessian=True)
    x = tokens[0].float()
    y = model[0](tokens[0]).float()
    assert torch.equal(rows['0'],x[:2]) and torch.equal(rows['1'],y[:2])
    assert torch.equal(hessians['0'],x.T@x) and torch.equal(hessians['1'],y.T@y)
    assert counts == {'0':3,'1':3}
    assert maxima == {'0':float(x.abs().max()),'1':float(y.abs().max())}
