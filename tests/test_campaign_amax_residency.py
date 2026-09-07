"""Calibration maxima stay on device until the complete draw has run."""
import torch
from prismaquant.tessera_campaign import _collect_activations


def test_maxima_convert_to_host_once_per_unit_and_keep_python_nan_semantics(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(2,2,bias=False)
            self.second = torch.nn.Linear(2,2,bias=False)
        def forward(self,x):
            return self.first(x)+self.second(x)
    model = Model()
    batches = [torch.tensor([[1.,2.]]),torch.tensor([[float('nan'),99.]]),
               torch.empty(0,2),torch.tensor([[-4.,3.]])]
    scalar_reads = []
    original = torch.Tensor.item
    def item(value,*args,**kwargs):
        scalar_reads.append(value.shape)
        return original(value,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'item',item)
    _x,_h,counts,maxima = _collect_activations(model,['first','second'],batches,1,'cpu',want_hessian=False)
    # Python max(0, NaN) ignores the NaN batch, including its finite 99.
    assert maxima == {'first':4.,'second':4.}
    assert counts == {'first':3,'second':3}
    assert scalar_reads == [torch.Size([]),torch.Size([])]
