#!/usr/bin/env bash
set -euo pipefail
image_name=prismaquant-qwen38-producer:20260827-tf516-hf128
docker inspect "$image_name" --format '{{.Id}}'
exec docker run --rm --gpus all --ipc=host \
  --mount "type=bind,src=$PWD,dst=/workspace,readonly" \
  --mount type=bind,src=/mnt/shared/prismaquant-validation/stack-producer-ba582d,dst=/producer,readonly \
  --workdir /workspace --entrypoint /bin/bash \
  --env PYTHONPATH=/workspace:/producer/src --env TESSERA_REPO=/producer \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  "$image_name" -c 'python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())" && python -m pytest -q -rs -p no:cacheprovider \
  tests/test_tessera_campaign.py::test_no_embedded_axis_on_the_wire \
  tests/test_tessera_campaign.py::test_the_priced_render_is_the_decoded_wire \
  tests/test_tessera_campaign.py::test_the_same_weight_rate_costs_differently_on_the_two_routes \
  tests/test_tessera_campaign.py::test_weights_only_is_reachable_but_only_deliberately \
  tests/test_tessera_campaign.py::test_weights_only_on_the_production_seam_is_a_stamped_lever'
