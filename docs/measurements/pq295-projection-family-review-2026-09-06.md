# Expert projection family alignment review

Reviewed base: PrismaQuant `c488b592` (2026-09-06 changes). Issue #295.

The existing retry sequence applied the same family-menu index to every expert
stack. Two legal, different menus `[E2M1x2, E4M3]` and `[E4M3, BF16]` never
produced the all-E4M3 request accepted by the current producer. The new
regression creates two real small unpacked BF16-source stacks, calls the
actual producer plan tool, binds all projected units, and compares their
source bytes with the live weights. Producer routes and projection results
are not mocked.

The repair preserves the existing nominal request sequence and adds any
missing request for a family shared by all stacks, matched by family name.
The request count is at most twice the largest distinct-family menu. No
per-stack subprocesses or Cartesian enumeration are introduced. It does not
claim exhaustive search for arbitrary mixed-family plans on a hypothetical
producer with different route support per stack. The current producer's
routed family support is build-wide and E4M3-only.

All execution below used published PrismaBuild. CPU tests reserved one CPU
and 4 GiB, hid CUDA, and bounded OMP/MKL/OpenBLAS threads to one. The producer
and test dependencies were the existing GB10 environment:
`/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python`,
`TESSERA_REPO=/home/rob/tessera`. These are small source/projection regressions,
not a full-model quantization, serving or performance qualification.

| Check | PB action | Actual result |
| --- | --- | --- |
| Before fix, new two-stack regression | `11355529964ff001fda25c2096465b85ff84f75144a927a91b711a5b8d04b703` | sparky, exit 1; 1 failed, 15 deselected, no skips. Producer refuses NVFP4 on the first stack, then BF16 on the second. |
| After fix, existing one-stack and new two-stack projection regressions | `72ae634c60923445547eb68a37267cdad780d8876051295961660727852ce672` | sparklina, exit 0; 2 passed, 33 deselected, no skips. The `-k` selector also deselected the initially supplied documentation tests; their separate action is below. |
| Architecture/staleness | `bdd55f472f52c60172d69b7500ded5b04c2bd4419762a776dcb54f236419bcda` | sparklina, exit 0; 19 passed, no skips. |
| Compile touched production/test modules | `304c94bedbe9c618d49c03d628ba4a8349000e0dd4319c53a95ca30684f8d06e` | sparklina, exit 0; 2 modules compiled. Portable standard Python, one CPU/1 GiB. |

Commands after the common `pbrun.py --cwd <review-checkout>` prefix:

```bash
--tag gb10 --cpus 1 --demand mem_gb=4 \
 --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
 --env TESSERA_REPO=/home/rob/tessera --timeout-s 300 -- \
 /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m pytest -q \
 tests/test_tessera_campaign_packed.py -k projection_matches_family_names

# Green projection action used -k 'projection_walks or projection_matches'.
# Documentation action used these paths without -k:
# tests/test_docs_staleness.py tests/test_architecture_doc.py
```

Terminal records and actual worker stdout were read. Successful CAS payloads
were read and SHA-256 checked independently:

| Action prefix | Receipt SHA-256 | Result SHA-256 |
| --- | --- | --- |
| `72ae634c6092` | `5cb0fb820fd852aa34dd77fcb478ab3da325c605e22b1b0f723fe2efcc870265` | `25213a684e5082fcf035ced399ed9f936506b00b2cbb11c3ed079addc4350e61` |
| `bdd55f472f52` | `a06a4e65243c379220b1e965ad86a01869750d61ed86d76b087e40ec59c90141` | `f04563fb40fd4b2b7d98fc2a0655558c112d653da161467187a739528a051cb0` |
| `304c94bedbe9` | `16c0fd9e3a31f8c020afce3fc19c02105612d81d0b6ca4405e2feddd30a8adc3` | `79cf46efdbff419748bb1683bba58a6c76cb507f0616a846b8de990491ba62c8` |

Terminal files live in `/mnt/shared/prismabuild-fleet/pb-queue/{done,failed}/<action>.json`; worker logs are under `attempts/<action>/`; result payloads are `/mnt/shared/prismabuild-fleet/cas/blobs/<first-two>/<result-sha256>`. No successful CAS receipt is claimed for the intentional RED. No GPU, quality, latency or work-per-joule delta is claimed.
