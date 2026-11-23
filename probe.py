#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path

import torch
from exllamav3 import Config, Model, Cache, Tokenizer, GreedySampler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model_dir = Path(args.model)
    config = Config.from_directory(str(model_dir))
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=2048)
    t0 = time.perf_counter()
    model.load(progressbar=True, device="cuda:0")
    load_s = time.perf_counter() - t0
    tokenizer = Tokenizer.from_config(config)
    prompt = "Write one short factual sentence about the Moon."
    ids = tokenizer.encode(prompt, encode_special_tokens=True)
    if ids.shape[-1] < 2:
        raise RuntimeError("tokenizer produced too few tokens")
    params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, 2048)}
    t0 = time.perf_counter()
    model.prefill(input_ids=ids[:, :-1], params=params)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0
    recurrent_states = params.get("recurrent_states")
    sampler = GreedySampler()
    vocab = tokenizer.get_id_to_piece_list()
    eos = set(getattr(config, "eos_token_id_list", []) or [])
    out = []
    generated = 0
    t0 = time.perf_counter()
    while generated < args.tokens:
        params = {
            "attn_mode": "flash_attn",
            "cache": cache,
            "past_len": ids.shape[-1] - 1,
            "batch_shape": (1, 2048),
            "recurrent_states": recurrent_states,
        }
        logits = model.forward(input_ids=ids[:, -1:], params=params)
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite logits")
        sample = sampler.forward(logits, tokenizer=tokenizer)
        token_id = int(sample.item())
        if token_id in eos:
            break
        ids = torch.cat((ids, sample.cpu()), dim=-1)
        if 0 <= token_id < len(vocab):
            out.append(vocab[token_id])
        generated += 1
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t0
    if recurrent_states:
        for rs in recurrent_states:
            rs.free()
    result = {
        "passed": generated > 0 and decode_s > 0,
        "model": str(model_dir),
        "load_s": round(load_s, 3),
        "prompt_tokens": int(ids.shape[-1] - generated),
        "prefill_s": round(prefill_s, 4),
        "generated_tokens": generated,
        "decode_s": round(decode_s, 4),
        "decode_tps": round(generated / decode_s, 3) if decode_s else None,
        "output_preview": "".join(out)[:500],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(3)

if __name__ == "__main__":
    main()
