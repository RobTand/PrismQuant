"""Full measured-wire census rows cannot invent unencoded stack rungs."""
from types import SimpleNamespace

from test_tessera_stack_sample_cost import STEM, _anchor, _packed_probe_row, _sample


def test_wire_backed_census_keeps_only_encoded_rungs_but_sample_still_estimates():
    from prismaquant import tessera_campaign as campaign

    family = "TESSERA_E2M1_K2"
    fmt = lambda rate: f"{family}_R{rate}"
    census = _sample(campaign, _packed_probe_row(2, [1.0, 3.0]))
    sample = _sample(campaign, _packed_probe_row(4, [1.0, 3.0, 2.0, 4.0]),
                     sampled=[0, 1], pi={0: 0.5, 1: 0.5})
    members = [name for names in census.members.values() for name in names]
    anchors = {name: {family: [
        _anchor(campaign, name, family, fmt(768), 768, 0.04),
        _anchor(campaign, name, family, fmt(1024), 1024, 0.01)]}
        for name in members}
    menu = [SimpleNamespace(family=family, format_name=fmt(rate), body_rate_q256=rate,
                            admission=SimpleNamespace(activation_contract="bfloat16"))
            for rate in (768, 896, 1024)]

    def payload(draw, backed):
        return campaign.campaign_cost_payload(
            anchors, {name: menu for name in members}, loo={}, provenance={},
            stack_samples={draw.packed_qname: draw}, wire_backed=backed)

    assert fmt(896) in payload(census, frozenset())["costs"][census.packed_qname]
    assert fmt(896) in payload(sample, frozenset(members))["costs"][sample.packed_qname]
    backed = payload(census, frozenset(members))["costs"][census.packed_qname]
    assert set(backed) == {fmt(768), fmt(1024)}
