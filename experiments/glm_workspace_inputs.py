"""Bind read-only source-prefix inputs for a PB workspace experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path


def fingerprint(path):
    stat = path.stat()
    # Shared NFS inode/ctime/mtime identify this object across eligible hosts;
    # local mount device numbers deliberately do not enter a portable binding.
    return dict(path=str(path.resolve()), inode=stat.st_ino, bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    from prismaquant.model_profiles import detect_profile
    from prismaquant.tessera_campaign import _calibration_tokens
    profile = detect_profile(str(args.model))
    assert profile.name == 'glm5_next'
    index = json.loads((args.model/'model.safetensors.index.json').read_text())['weight_map']
    prefix = 'model.language_model.layers.'
    shards = set()
    for key, shard in index.items():
        live = profile.checkpoint_to_live_name(key, multimodal=True)
        if live is None:
            continue
        if live.startswith(prefix) and int(live[len(prefix):].split('.')[0]) > args.layer+1:
            continue
        shards.add(args.model/shard)
    files = sorted(shards | {p for p in args.model.iterdir()
                            if p.is_file() and p.suffix in ('.json', '.model', '.jinja')})

    def bind(path):
        before = fingerprint(path)
        with path.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert fingerprint(path) == before, f'source changed while hashing {path}'
        return dict(**before, sha256=sha)

    with ThreadPoolExecutor(max_workers=4) as pool:
        bindings = list(pool.map(bind, files))
    (args.out/'source-files.json').write_text(json.dumps(bindings, indent=2)+'\n')
    tokens, corpus = _calibration_tokens(str(args.model), 512, 512, 0)
    assert len(tokens) == 512 and all(tuple(t.shape) == (1, 512) for t in tokens)
    token_bytes = json.dumps([t.tolist() for t in tokens], separators=(',', ':')).encode()
    (args.out/'tokens.json').write_bytes(token_bytes)
    payload = dict(schema='prismaquant.glm_workspace_inputs.v1',
        scope='bounded_real_source_prefix_workspace_only_unfrozen_production_draw',
        model=str(args.model), stop_after_layer=args.layer, nsamples=512, seqlen=512,
        original_microbatch=1, seed=0, source_files=bindings,
        corpus_sha256=hashlib.sha256(corpus.encode()).hexdigest(),
        tokens_path=str(args.out/'tokens.json'), tokens_sha256=hashlib.sha256(token_bytes).hexdigest())
    (args.out/'binding.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(dict(binding_path=str(args.out/'binding.json'),
        binding_sha256=hashlib.sha256((args.out/'binding.json').read_bytes()).hexdigest(),
        source_files=len(bindings), source_bytes=sum(x['bytes'] for x in bindings),
        tokens_sha256=payload['tokens_sha256']), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
