from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from prismaquant.cluster_campaign import (
    seal_campaign_manifest_v2,
    sealed_stage_receipt_sha256,
    validate_campaign_manifest_v2,
)


_TOOL = Path(__file__).resolve().parents[1] / "tools" / (
    "build_prismasnap_27b_campaign.py"
)
_SPEC = importlib.util.spec_from_file_location("prismasnap_27b_campaign", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)
_COMMIT = "a" * 40


def _stages(body):
    return {row["id"]: row for row in body["stages"]}


def _child(stage):
    marker = stage["argv"].index("--")
    return stage["argv"][marker + 1 :]


def test_campaign_body_is_deterministic_sealed_v2_with_exact_two_host_dag():
    first = builder.build_campaign_body(_COMMIT)
    second = builder.build_campaign_body(_COMMIT)
    assert first == second
    manifest = seal_campaign_manifest_v2(first)
    assert validate_campaign_manifest_v2(manifest) == manifest
    assert seal_campaign_manifest_v2(second)["identity_sha256"] == manifest[
        "identity_sha256"
    ]
    assert first["coordinator"] == "sparky"
    assert first["max_parallel"] == 2
    assert {row["id"] for row in first["hosts"]} == {"sparky", "sparklina"}
    remote = next(row for row in first["hosts"] if row["id"] == "sparklina")
    transport = remote["transport"]
    assert transport["host"] == "10.100.96.2"
    assert transport["ssh_executable"] == str(builder.SSH_SHIM)
    assert transport["remote_helper_argv"][-1] == "_exec-request"

    stages = _stages(first)
    assert set(stages) == {
        "preflight-sparky",
        "preflight-sparklina",
        "bind-probe",
        "scan-tensor-metadata",
        "push-planning-inputs",
        "plan-sparky",
        "plan-sparklina",
        "fetch-plan-sparklina",
        "merge-plans",
        "push-merged-plan",
        "materialize-part-sparky",
        "materialize-part-sparklina",
        "fetch-part-sparklina",
        "merge-checkpoint-parts",
    }
    assert stages["bind-probe"]["dependencies"] == [
        "preflight-sparky",
        "preflight-sparklina",
    ]
    assert stages["scan-tensor-metadata"]["dependencies"] == ["bind-probe"]
    assert stages["push-planning-inputs"]["dependencies"] == [
        "scan-tensor-metadata"
    ]
    assert stages["fetch-plan-sparklina"]["dependencies"] == [
        "plan-sparky",
        "plan-sparklina",
    ]
    assert stages["fetch-part-sparklina"]["dependencies"] == [
        "materialize-part-sparky",
        "materialize-part-sparklina",
    ]

    for stage_id, stage in stages.items():
        child = _child(stage)
        token = f"{builder.CAMPAIGN_ID}:{stage_id}"
        assert stage["receipts"] == [
            {
                "path": str(
                    builder.RUN_ROOT
                    / "campaign"
                    / "receipts"
                    / f"{stage_id}.sealed.json"
                ),
                "sha256": sealed_stage_receipt_sha256(token, child),
            }
        ]


def test_planners_use_fast_exact_defaults_external_census_and_hf_alias():
    stages = _stages(builder.build_campaign_body(_COMMIT))
    for host_id, image, layers in (
        ("sparky", builder.LOCAL_IMAGE_ID, "0-31"),
        ("sparklina", builder.REMOTE_IMAGE_ID, "32-63"),
    ):
        argv = _child(stages[f"plan-{host_id}"])
        assert argv[0:3] == ["/usr/bin/docker", "run", "--rm"]
        assert image in argv
        assert ["--gpus", "all"] == argv[
            argv.index("--gpus") : argv.index("--gpus") + 2
        ]
        assert argv[argv.index("--source") + 1] == str(builder.HF_ALIAS)
        assert str(builder.SOURCE_ROOT) not in argv[argv.index("--source") + 1 :]
        assert argv[argv.index("--layers") + 1] == layers
        assert argv[argv.index("--tensor-metadata-manifest") + 1] == str(
            builder.RUN_ROOT / "probe" / "tensor-metadata-manifest.json"
        )
        assert argv[argv.index("--alphas") + 1 : argv.index("--max-rounds")] == [
            "0.0",
            "0.125",
            "0.25",
            "0.375",
            "0.5",
        ]
        assert argv[argv.index("--max-rounds") + 1] == "4"
        assert argv[argv.index("--polish-top") + 1] == "8"
        assert argv[argv.index("--polish-pool") + 1] == "16"
        assert argv[argv.index("--nvfp4-scale-rule") + 1] == "static_6"
        assert "--skip-source-content-verification" not in argv

    bind = _child(stages["bind-probe"])
    scan = _child(stages["scan-tensor-metadata"])
    assert bind[bind.index("--source") + 1] == str(builder.HF_ALIAS)
    assert scan[scan.index("--source") + 1] == str(builder.HF_ALIAS)
    assert "--gpus" not in bind
    assert "--gpus" not in scan


def test_materialization_is_parallel_split_and_final_merge_requires_hardlinks():
    stages = _stages(builder.build_campaign_body(_COMMIT))
    for host_id, image in (
        ("sparky", builder.LOCAL_IMAGE_ID),
        ("sparklina", builder.REMOTE_IMAGE_ID),
    ):
        argv = _child(stages[f"materialize-part-{host_id}"])
        assert image in argv
        assert "--gpus" in argv
        assert argv[argv.index("--shards-file") + 1] == str(
            builder.RUN_ROOT / "inputs" / f"shards-{host_id}.txt"
        )
        assert argv[argv.index("--device") + 1] == "cuda"
    final = _child(stages["merge-checkpoint-parts"])
    assert "--require-hardlinks" in final
    assert "--gpus" not in final
    assert final.count("--part") == 2
    assert final[final.index("--output") + 1] == str(
        builder.RUN_ROOT / "snapped-source"
    )


def test_transfer_stages_are_content_receipted_and_rsync_is_checksum_pinned():
    stages = _stages(builder.build_campaign_body(_COMMIT))
    expected = {
        "push-planning-inputs": "push",
        "fetch-plan-sparklina": "fetch",
        "push-merged-plan": "push",
        "fetch-part-sparklina": "fetch",
    }
    for stage_id, direction in expected.items():
        argv = _child(stages[stage_id])
        assert argv[2].endswith("build_prismasnap_27b_campaign.py")
        assert argv[3] == "transfer-tree"
        assert argv[argv.index("--direction") + 1] == direction
        assert argv[argv.index("--receipt") + 1].endswith(f"/{stage_id}.json")

    rsync = builder._rsync_argv(
        direction="push", source=Path("/abs/source"), destination=Path("/abs/dest")
    )
    assert rsync[:2] == ["/usr/bin/rsync", "--archive"]
    for flag in (
        "--checksum",
        "--partial",
        "--protect-args",
        "--mkpath",
        "--rsync-path=/usr/bin/rsync",
        f"--rsh={builder.SSH_SHIM}",
    ):
        assert flag in rsync


def test_preflight_pins_images_gpus_disk_source_and_nonambient_ssh():
    stages = _stages(builder.build_campaign_body(_COMMIT))
    for host_id in ("sparky", "sparklina"):
        argv = _child(stages[f"preflight-{host_id}"])
        assert argv[-4:] == [
            "--host",
            host_id,
            "--producer-git-commit",
            _COMMIT,
        ]
    assert builder._host_config("sparky") == (
        builder.LOCAL_IMAGE_ID,
        builder.LOCAL_GPU_UUID,
        270_000_000_000,
    )
    assert builder._host_config("sparklina") == (
        builder.REMOTE_IMAGE_ID,
        builder.REMOTE_GPU_UUID,
        135_000_000_000,
    )
    shim = builder._ssh_shim_bytes().decode("utf-8")
    for pin in (
        "-F",
        "/dev/null",
        str(builder.SSH_KEY),
        "BatchMode=yes",
        "StrictHostKeyChecking=yes",
        f"UserKnownHostsFile={builder.KNOWN_HOSTS}",
        "ConnectTimeout=20",
    ):
        assert pin in shim


def test_tree_manifest_is_content_deterministic_and_rejects_symlink(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "b").write_bytes(b"two")
    (root / "a").write_bytes(b"one")
    first = builder._tree_manifest(root)
    second = builder._tree_manifest(root)
    assert first == second
    assert [row["path"] for row in first["files"]] == ["a", "b"]
    (root / "link").symlink_to(root / "a")
    with pytest.raises(builder.CampaignBuildError, match="symlink"):
        builder._tree_manifest(root)


def test_rootfs_digest_algorithm_is_compact_sorted_json_with_trailing_lf():
    layers = ["sha256:b", "sha256:a"]
    expected = hashlib.sha256(b'["sha256:b","sha256:a"]\n').hexdigest()
    assert builder._rootfs_layers_sha256(layers) == expected


def test_commit_must_be_exact_full_lowercase_hex():
    for value in ("a" * 39, "A" * 40, "g" * 40):
        with pytest.raises(builder.CampaignBuildError, match="commit"):
            builder.build_campaign_body(value)


def test_hash_only_bootstrap_does_not_import_torch_or_package_initializer(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "payload").write_bytes(b"content")
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    (blocker / "torch.py").write_text(
        "raise RuntimeError('torch must not be imported by bootstrap')\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(_TOOL), "hash-tree", "--path", str(tree)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PYTHONPATH": str(blocker), "PATH": "/usr/bin:/bin"},
    )
    assert completed.returncode == 0, completed.stderr
    assert '"sha256"' in completed.stdout
