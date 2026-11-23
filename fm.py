#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PKG = Path(__file__).resolve().parent
CFG = {
    "workspace_root": str(Path.home() / "exl3-first-mover-work"),
    "exllamav3_repo": os.environ.get("EXLLAMAV3_REPO", str(Path.home() / "exllamav3")),
    "default_devices": [0],
    "sweet_min_b": 3.0,
    "sweet_max_b": 32.0,
    "hard_max_b": 40.0,
    "max_source_gib": 70.0,
    "default_bits": 4.0,
    "shard_size_mb": 4096,
    "probe_tokens": 64,
    "disk_headroom_gib": 15.0,
    "tabby_repo": os.environ.get("TABBYAPI_REPO", str(Path.home() / "tabbyAPI")),
    "tabby_required_for_publish": True,
    "tabby_context": 8192,
    "tabby_port": 5011,
    "tabby_max_batch_size": 2
}
local_cfg = PKG / "config.json"
if local_cfg.exists():
    CFG.update(json.loads(local_cfg.read_text()))


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def slugify(repo_id):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", repo_id.replace("/", "__"))

def workdir(repo_id):
    return Path(CFG["workspace_root"]) / slugify(repo_id)

def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default

def run(cmd, cwd=None, log=None, check=True):
    print("+", " ".join(map(str, cmd)), flush=True)
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with Path(log).open("a", buffering=1) as f:
            cp = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        cp = subprocess.run(cmd, cwd=cwd, text=True)
    if check and cp.returncode:
        raise RuntimeError(f"command failed ({cp.returncode}): {' '.join(map(str, cmd))}")
    return cp

def hf():
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except Exception as e:
        raise RuntimeError("huggingface_hub missing; run ./run.sh install") from e
    return HfApi, hf_hub_download, snapshot_download

def get_card_license(info):
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    if isinstance(card, dict):
        return card.get("license")
    return getattr(card, "license", None)

def get_total_params(info):
    st = getattr(info, "safetensors", None)
    if st is None:
        return None
    if isinstance(st, dict):
        return st.get("total")
    return getattr(st, "total", None)

def source_weight_bytes(info):
    total = 0
    for s in info.siblings:
        if s.rfilename.endswith(".safetensors"):
            total += int(getattr(s, "size", 0) or 0)
    return total

def config_is_moe(cfg):
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    keys = ("num_experts", "num_local_experts", "n_routed_experts", "num_experts_per_tok")
    return any(int(text.get(k, 0) or 0) > 0 for k in keys)

def architecture_supported(arch):
    repo = Path(CFG["exllamav3_repo"])
    root = repo / "exllamav3" / "architecture"
    if not root.exists():
        return None
    try:
        for p in root.glob("*.py"):
            if arch in p.read_text(errors="ignore"):
                return True
        return False
    except Exception:
        return None

def competition(api, repo_id):
    base = repo_id.split("/")[-1]
    found = []
    try:
        for m in api.list_models(search=base, limit=50):
            mid = getattr(m, "id", "") or getattr(m, "modelId", "")
            low = mid.lower()
            if mid != repo_id and ("exl3" in low or "exllamav3" in low):
                found.append({
                    "repo_id": mid,
                    "downloads": getattr(m, "downloads", None),
                    "likes": getattr(m, "likes", None),
                })
    except Exception as e:
        return [{"scan_error": str(e)}]
    return found[:20]

def preflight(repo_id, job=None):
    HfApi, hf_hub_download, _ = hf()
    api = HfApi()
    info = api.model_info(repo_id, files_metadata=True)
    wd = workdir(repo_id)
    meta = wd / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    cfg_path = hf_hub_download(repo_id, "config.json", local_dir=meta)
    model_cfg = json.loads(Path(cfg_path).read_text())
    text_cfg = model_cfg.get("text_config") if isinstance(model_cfg.get("text_config"), dict) else model_cfg
    archs = model_cfg.get("architectures") or text_cfg.get("architectures") or []
    arch = archs[0] if archs else ""
    wb = source_weight_bytes(info)
    source_gib = wb / (1024 ** 3)
    params = get_total_params(info)
    param_b = params / 1e9 if isinstance(params, int) and params > 0 else wb / 2e9 if wb else None
    moe = config_is_moe(model_cfg)
    allow_large = bool((job or {}).get("allow_large", False))
    sweet = "GREEN"
    reasons = []
    if param_b is not None and param_b < float(CFG["sweet_min_b"]):
        sweet = "AMBER"
        reasons.append("below preferred first-mover size")
    if param_b is not None and param_b > float(CFG["sweet_max_b"]):
        sweet = "AMBER"
        reasons.append("above preferred 32B sweet spot")
    if source_gib > float(CFG["max_source_gib"]) or (param_b is not None and param_b > float(CFG["hard_max_b"])):
        sweet = "OVERRIDE" if allow_large else "BLOCKED"
        reasons.append("too large for fast local first-mover workflow")
    targets = (job or {}).get("targets") or [CFG["default_bits"]]
    out_est_gib = source_gib * (float(max(targets)) / 16.0)
    need_gib = source_gib + out_est_gib * 2.2 + float(CFG["disk_headroom_gib"])
    free_gib = shutil.disk_usage(Path(CFG["workspace_root"]).parent).free / (1024 ** 3)
    if free_gib < need_gib:
        sweet = "BLOCKED"
        reasons.append(f"disk budget requires about {need_gib:.1f} GiB, free {free_gib:.1f} GiB")
    result = {
        "time": now(),
        "repo_id": repo_id,
        "source_revision": getattr(info, "sha", None),
        "architecture": arch,
        "model_type": text_cfg.get("model_type") or model_cfg.get("model_type"),
        "source_weight_gib": round(source_gib, 3),
        "parameters_b": round(param_b, 3) if param_b is not None else None,
        "moe": moe,
        "license": get_card_license(info),
        "private": bool(getattr(info, "private", False)),
        "gated": getattr(info, "gated", False),
        "exllamav3_arch_supported": architecture_supported(arch),
        "existing_exl3": competition(api, repo_id),
        "targets": targets,
        "estimated_disk_gib": round(need_gib, 2),
        "free_disk_gib": round(free_gib, 2),
        "sweet_spot": sweet,
        "reasons": reasons,
    }
    dump(wd / "preflight.json", result)
    print(json.dumps(result, indent=2))
    return result

def doctor():
    data = {
        "time": now(),
        "python": sys.version.split()[0],
        "workspace_root": CFG["workspace_root"],
        "exllamav3_repo": CFG["exllamav3_repo"],
        "convert_py": str(Path(CFG["exllamav3_repo"]) / "convert.py"),
        "convert_exists": (Path(CFG["exllamav3_repo"]) / "convert.py").exists(),
        "venv": str(PKG / ".venv"),
        "venv_exists": (PKG / ".venv" / "bin" / "python").exists(),
    }
    try:
        import torch
        data["torch"] = torch.__version__
        data["cuda_available"] = torch.cuda.is_available()
        data["cuda"] = torch.version.cuda
        data["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as e:
        data["torch_error"] = str(e)
    try:
        HfApi, _, _ = hf()
        data["hf_auth"] = bool(HfApi().whoami())
    except Exception as e:
        data["hf_auth"] = False
        data["hf_note"] = str(e)
    print(json.dumps(data, indent=2))

def download_source(repo_id, revision):
    _, _, snapshot_download = hf()
    wd = workdir(repo_id)
    src = wd / "source"
    src.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id,
        revision=revision,
        local_dir=src,
        ignore_patterns=["*.bin", "*.pt", "*.pth", "*.gguf", "*.onnx", "optimizer*", "training_args*"],
    )
    return src


def normalize_qwen35_processor(repo_id, source, pf):
    arch = pf.get("architecture")
    if arch not in {"Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration", "Qwen3_5Model"}:
        return []
    changes = []
    backup = workdir(repo_id) / "meta" / "source-original-metadata"
    pre = source / "preprocessor_config.json"
    if pre.exists():
        d = json.loads(pre.read_text())
        if d.get("image_processor_type") == "Qwen2VLImageProcessor":
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pre, backup / pre.name)
            d["image_processor_type"] = "Qwen2VLImageProcessorFast"
            d.setdefault("processor_class", "Qwen3VLProcessor")
            pre.write_text(json.dumps(d, indent=2) + "\n")
            changes.append({
                "file": pre.name,
                "field": "image_processor_type",
                "from": "Qwen2VLImageProcessor",
                "to": "Qwen2VLImageProcessorFast",
                "reason": "ExLlamaV3 Qwen3.5 loader requires the Fast processor metadata; official Qwen3.5 checkpoints use this value with the same processor geometry."
            })
    proc = source / "processor_config.json"
    if proc.exists():
        d = json.loads(proc.read_text())
        ip = d.get("image_processor")
        if isinstance(ip, dict) and ip.get("image_processor_type") == "Qwen2VLImageProcessor":
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(proc, backup / proc.name)
            ip["image_processor_type"] = "Qwen2VLImageProcessorFast"
            proc.write_text(json.dumps(d, indent=2) + "\n")
            changes.append({
                "file": proc.name,
                "field": "image_processor.image_processor_type",
                "from": "Qwen2VLImageProcessor",
                "to": "Qwen2VLImageProcessorFast",
                "reason": "Keep processor metadata consistent with ExLlamaV3/Qwen3.5 expectations."
            })
        if not pre.exists() and isinstance(ip, dict):
            pp = dict(ip)
            pp["processor_class"] = d.get("processor_class", "Qwen3VLProcessor")
            pre.write_text(json.dumps(pp, indent=2) + "\n")
            changes.append({
                "file": pre.name,
                "field": "synthesized_from_processor_config.image_processor",
                "from": None,
                "to": "created",
                "reason": "ExLlamaV3 Qwen3.5 loader requires preprocessor_config.json even when the release only embeds image processor metadata in processor_config.json."
            })
    cfg_file = source / "config.json"
    if cfg_file.exists():
        cfg = json.loads(cfg_file.read_text())
        tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
        declared_mtp = int(tc.get("mtp_num_hidden_layers", 0) or 0)
        idx = source / "model.safetensors.index.json"
        keys = []
        if idx.exists():
            try:
                keys = list(json.loads(idx.read_text()).get("weight_map", {}))
            except Exception:
                keys = []
        observed_mtp = any(("mtp" in k.lower()) or ("nextn" in k.lower()) for k in keys)
        if declared_mtp > 0 and keys and not observed_mtp:
            backup.mkdir(parents=True, exist_ok=True)
            if not (backup / cfg_file.name).exists():
                shutil.copy2(cfg_file, backup / cfg_file.name)
            tc["mtp_num_hidden_layers"] = 0
            cfg_file.write_text(json.dumps(cfg, indent=2) + "\n")
            changes.append({
                "file": cfg_file.name,
                "field": "text_config.mtp_num_hidden_layers" if tc is not cfg else "mtp_num_hidden_layers",
                "from": declared_mtp,
                "to": 0,
                "reason": "Source config declares MTP/NextN layers, but the safetensors index contains no MTP/NextN tensors; disable the absent side model for EXL3 conversion."
            })
    if changes:
        np = workdir(repo_id) / "source-normalizations.json"
        prior = read_json(np, {}) or {}
        merged = prior.get("changes", []) if isinstance(prior.get("changes", []), list) else []
        seen = {(x.get("file"), x.get("field"), str(x.get("from")), str(x.get("to"))) for x in merged if isinstance(x, dict)}
        for x in changes:
            key = (x.get("file"), x.get("field"), str(x.get("from")), str(x.get("to")))
            if key not in seen:
                merged.append(x)
                seen.add(key)
        dump(np, {"time": now(), "changes": merged})
    return changes

def copy_tokenizer_files(source, out):
    names = [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "tokenizer.model", "chat_template.jinja", "generation_config.json",
        "preprocessor_config.json", "processor_config.json",
        "joint_head.safetensors", "joint_head_config.json", "joint_schema_model.py", "LICENSE",
        "readout.safetensors", "decision_config.json", "inference.py",
    ]
    for name in names:
        p = source / name
        q = out / name
        if p.exists() and not q.exists():
            shutil.copy2(p, q)

def quantize(repo_id, source, bits, job, pf):
    wd = workdir(repo_id)
    exl_repo = Path(CFG["exllamav3_repo"])
    if not (exl_repo / "convert.py").exists():
        raise RuntimeError("ExLlamaV3 checkout missing; run ./run.sh install")
    label = f"{float(bits):.2f}bpw"
    out = wd / "artifacts" / label
    exlw = wd / "exl-work" / label
    state = wd / "state" / f"{label}.json"
    sig = {
        "repo_id": repo_id,
        "revision": pf["source_revision"],
        "bits": float(bits),
        "devices": job.get("devices") or CFG["default_devices"],
        "hq": job.get("hq", "auto"),
        "shard_size_mb": int(job.get("shard_size_mb", CFG["shard_size_mb"])),
        "head_bits": job.get("head_bits"),
        "mtp_bits": job.get("mtp_bits"),
        "vision_bits": job.get("vision_bits"),
        "ngram_bits": job.get("ngram_bits"),
    }
    old = read_json(state)
    if old and old != sig:
        raise RuntimeError(f"job signature changed for {label}; remove {exlw} explicitly before retrying")
    dump(state, sig)
    out.mkdir(parents=True, exist_ok=True)
    exlw.mkdir(parents=True, exist_ok=True)
    complete = (out / "config.json").exists() and any(out.glob("*.safetensors"))
    if complete:
        return out
    py = str(PKG / ".venv" / "bin" / "python") if (PKG / ".venv" / "bin" / "python").exists() else sys.executable
    if any(exlw.iterdir()):
        cmd = [py, str(PKG / "exl_convert.py"), "-w", str(exlw), "-r"]
    else:
        cmd = [
            py, str(PKG / "exl_convert.py"),
            "-i", str(source),
            "-o", str(out),
            "-w", str(exlw),
            "-b", str(float(bits)),
            "-ss", str(sig["shard_size_mb"]),
            "-d", ",".join(str(x) for x in sig["devices"]),
        ]
        hq = sig["hq"]
        if hq is True or (hq == "auto" and pf["moe"]):
            cmd.append("-hq")
        if sig["head_bits"] is not None:
            cmd += ["-hb", str(sig["head_bits"])]
        if sig["mtp_bits"] is not None:
            cmd += ["-mb", str(sig["mtp_bits"])]
        if sig["vision_bits"] is not None:
            cmd += ["-vb", str(sig["vision_bits"])]
        if sig["ngram_bits"] is not None:
            cmd += ["-ngb", str(sig["ngram_bits"])]
    run(cmd, cwd=PKG, log=wd / "logs" / f"convert-{label}.log")
    copy_tokenizer_files(source, out)
    if not (out / "config.json").exists() or not any(out.glob("*.safetensors")):
        raise RuntimeError(f"conversion finished without a complete artifact: {out}")
    return out

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def manifest_dir(path):
    files = {}
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.name != "README.md":
            files[str(p.relative_to(path))] = {"size": p.stat().st_size, "sha256": sha256(p)}
    return files

def exl3_revision():
    try:
        from importlib.metadata import version
        return version("exllamav3")
    except Exception:
        return None

def write_card(repo_id, bits, out, pf, probe):
    license_id = pf.get("license") or "other"
    yaml = f"---\nbase_model: {repo_id}\nlicense: {license_id}\ntags:\n- exl3\n- exllamav3\n- quantized\n---\n"
    text = yaml + f"""
# {repo_id.split("/")[-1]} EXL3 {float(bits):.2f} bpw

Source: https://huggingface.co/{repo_id}

Source revision: `{pf.get("source_revision")}`

ExLlamaV3 revision: `{exl3_revision()}`

Target bitrate: **{float(bits):.2f} bpw**

Architecture: `{pf.get("architecture")}`

This conversion was produced with ExLlamaV3's official `convert.py` and passed a local load/generation smoke test before publication.

## Local validation

- Generated tokens: {probe.get("generated_tokens")}
- Decode speed: {probe.get("decode_tps")} tok/s
- GPU: {probe.get("gpu")}
- Torch: {probe.get("torch")}

The speed figure is a local smoke benchmark, not a cross-system performance claim.
"""
    (out / "README.md").write_text(text.strip() + "\n")

def tabby_smoke(repo_id, bits, out, job):
    wd = workdir(repo_id)
    label = f"{float(bits):.2f}bpw"
    result_path = wd / f"tabby-smoke-{label}.json"
    log_path = wd / "logs" / f"tabby-{label}.log"
    py = str(PKG / ".venv" / "bin" / "python") if (PKG / ".venv" / "bin" / "python").exists() else sys.executable
    cmd = [
        py, str(PKG / "tabby_smoke.py"),
        "--model", str(out),
        "--tabby-repo", str(job.get("tabby_repo", CFG["tabby_repo"])),
        "--port", str(int(job.get("tabby_port", CFG["tabby_port"]))),
        "--context", str(int(job.get("tabby_context", CFG["tabby_context"]))),
        "--max-batch", str(int(job.get("tabby_max_batch_size", CFG["tabby_max_batch_size"]))),
        "--out", str(result_path),
        "--log", str(log_path),
    ]
    run(cmd, cwd=PKG, log=wd / "logs" / f"tabby-run-{label}.log")
    data = read_json(result_path)
    if not data or not data.get("passed"):
        raise RuntimeError(f"TabbyAPI smoke failed for {label}")
    return data


def probe(repo_id, bits, out, tokens):
    wd = workdir(repo_id)
    label = f"{float(bits):.2f}bpw"
    result_path = wd / f"probe-{label}.json"
    py = str(PKG / ".venv" / "bin" / "python") if (PKG / ".venv" / "bin" / "python").exists() else sys.executable
    cmd = [py, str(PKG / "probe.py"), "--model", str(out), "--tokens", str(tokens), "--out", str(result_path)]
    run(
        cmd,
        log=wd / "logs" / f"probe-{label}.log",
    )
    result = read_json(result_path)
    if not result or not result.get("passed"):
        raise RuntimeError(f"probe failed for {label}")
    return result

def set_status(repo_id, state, **extra):
    wd = workdir(repo_id)
    old = read_json(wd / "status.json", {}) or {}
    keep = {k: old[k] for k in ("pid", "log") if k in old}
    data = {"repo_id": repo_id, "state": state, "updated": now(), **keep, **extra}
    dump(wd / "status.json", data)

def publish_one(repo_id, bits, dest_repo, allow_unknown_license=False):
    HfApi, _, _ = hf()
    api = HfApi()
    wd = workdir(repo_id)
    label = f"{float(bits):.2f}bpw"
    out = wd / "artifacts" / label
    pf = read_json(wd / "preflight.json")
    probe_data = read_json(wd / f"probe-{label}.json")
    mani = read_json(wd / f"manifest-{label}.json")
    if not pf or not probe_data or not mani:
        raise RuntimeError("publish gate missing preflight/probe/manifest")
    if not probe_data.get("passed"):
        raise RuntimeError("publish blocked: smoke probe did not pass")
    if CFG.get("tabby_required_for_publish", True) and pf.get("architecture") != "Qwen3_5Model":
        tabby_data = read_json(wd / f"tabby-smoke-{label}.json")
        if not tabby_data or not tabby_data.get("passed"):
            raise RuntimeError("publish blocked: TabbyAPI smoke did not pass")
    if not pf.get("license") and not allow_unknown_license:
        raise RuntimeError("publish blocked: source license unknown")
    current = manifest_dir(out)
    if current != mani.get("files"):
        raise RuntimeError("publish blocked: artifact changed after validation")
    api.create_repo(dest_repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=dest_repo,
        repo_type="model",
        folder_path=str(out),
        commit_message=f"Publish EXL3 {float(bits):.2f} bpw from {repo_id}",
    )
    url = f"https://huggingface.co/{dest_repo}"
    dump(wd / f"published-{label}.json", {"time": now(), "dest_repo": dest_repo, "url": url})
    print(url)
    return url

def run_job(job_path):
    job = json.loads(Path(job_path).read_text())
    repo_id = job["repo_id"]
    wd = workdir(repo_id)
    wd.mkdir(parents=True, exist_ok=True)
    dump(wd / "job.json", job)
    try:
        set_status(repo_id, "PREFLIGHT")
        pf = preflight(repo_id, job)
        if pf["sweet_spot"] == "BLOCKED":
            set_status(repo_id, "BLOCKED", reason="; ".join(pf["reasons"]))
            return 4
        if pf["exllamav3_arch_supported"] is False:
            set_status(repo_id, "BLOCKED", reason=f"architecture not found in local ExLlamaV3: {pf['architecture']}")
            return 4
        set_status(repo_id, "DOWNLOADING")
        source = download_source(repo_id, pf["source_revision"])
        normalizations = normalize_qwen35_processor(repo_id, source, pf)
        if normalizations:
            set_status(repo_id, "NORMALIZED_METADATA", normalizations=normalizations)
        results = []
        for bits in job.get("targets") or [CFG["default_bits"]]:
            label = f"{float(bits):.2f}bpw"
            set_status(repo_id, "QUANTIZING", current=label)
            out = quantize(repo_id, source, bits, job, pf)
            set_status(repo_id, "PROBING", current=label)
            p = probe(repo_id, bits, out, int(job.get("probe_tokens", CFG["probe_tokens"])))
            t = None
            if job.get("tabby_smoke", True):
                set_status(repo_id, "TABBY_SMOKE", current=label)
                t = tabby_smoke(repo_id, bits, out, job)
            write_card(repo_id, bits, out, pf, p)
            mani = {
                "time": now(),
                "repo_id": repo_id,
                "source_revision": pf["source_revision"],
                "exllamav3_revision": exl3_revision(),
                "bits": float(bits),
                "files": manifest_dir(out),
                "probe": p,
                "tabby_smoke": t,
            }
            dump(wd / f"manifest-{label}.json", mani)
            item = {"bits": float(bits), "artifact": str(out), "probe": p, "tabby_smoke": t}
            if job.get("publish"):
                dest = job.get("dest_repo")
                if not dest:
                    raise RuntimeError("publish requested but dest_repo is null")
                if len(job.get("targets") or []) > 1:
                    raise RuntimeError("automatic publish supports one target per job")
                item["published"] = publish_one(
                    repo_id,
                    bits,
                    dest,
                    bool(job.get("allow_unknown_license", False)),
                )
            results.append(item)
        set_status(repo_id, "DONE", results=results)
        return 0
    except Exception as e:
        set_status(repo_id, "FAILED", error=str(e))
        raise

def launch(job_path):
    job = json.loads(Path(job_path).read_text())
    repo_id = job["repo_id"]
    wd = workdir(repo_id)
    wd.mkdir(parents=True, exist_ok=True)
    log = wd / "logs" / "worker.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    py = str(PKG / ".venv" / "bin" / "python") if (PKG / ".venv" / "bin" / "python").exists() else sys.executable
    with log.open("ab") as f:
        p = subprocess.Popen(
            [py, str(PKG / "fm.py"), "job", str(Path(job_path).resolve())],
            cwd=PKG,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    dump(wd / "worker.json", {"pid": p.pid, "started": now(), "job": str(Path(job_path).resolve())})
    set_status(repo_id, "LAUNCHED", pid=p.pid, log=str(log))
    print(json.dumps({"repo_id": repo_id, "pid": p.pid, "status": str(wd / "status.json"), "log": str(log)}, indent=2))

def status(repo_id):
    wd = workdir(repo_id)
    s = read_json(wd / "status.json")
    if not s:
        print(json.dumps({"repo_id": repo_id, "state": "NOT_FOUND"}, indent=2))
        return
    slim = dict(s)
    results = slim.get("results")
    if results:
        slim["results"] = [
            {
                "bits": r.get("bits"),
                "artifact": r.get("artifact"),
                "decode_tps": (r.get("probe") or {}).get("decode_tps"),
                "gpu": (r.get("probe") or {}).get("gpu"),
                "published": r.get("published"),
            }
            for r in results
        ]
    print(json.dumps(slim, indent=2))

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    p = sub.add_parser("preflight")
    p.add_argument("repo_id")
    p = sub.add_parser("job")
    p.add_argument("job_json")
    p = sub.add_parser("launch")
    p.add_argument("job_json")
    p = sub.add_parser("status")
    p.add_argument("repo_id")
    p = sub.add_parser("publish")
    p.add_argument("repo_id")
    p.add_argument("--dest-repo", required=True)
    p.add_argument("--bits", type=float, default=4.0)
    p.add_argument("--allow-unknown-license", action="store_true")
    args = ap.parse_args()
    if args.cmd == "doctor":
        doctor()
    elif args.cmd == "preflight":
        preflight(args.repo_id, {})
    elif args.cmd == "job":
        raise SystemExit(run_job(args.job_json))
    elif args.cmd == "launch":
        launch(args.job_json)
    elif args.cmd == "status":
        status(args.repo_id)
    elif args.cmd == "publish":
        publish_one(args.repo_id, args.bits, args.dest_repo, args.allow_unknown_license)

if __name__ == "__main__":
    main()
