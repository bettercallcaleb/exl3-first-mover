#!/usr/bin/env python3
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image, ImageDraw

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tabby-repo", required=True)
    ap.add_argument("--port", type=int, default=5011)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--max-batch", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    args=ap.parse_args()

    model=Path(args.model).resolve()
    tabby=Path(args.tabby_repo).resolve()
    if not (tabby/"main.py").exists():
        raise RuntimeError(f"TabbyAPI not found: {tabby}")

    cfg=json.loads((model/"config.json").read_text())
    expect_vision=bool(cfg.get("vision_config"))
    template=(model/"chat_template.jinja").read_text(errors="ignore") if (model/"chat_template.jinja").exists() else ""
    expect_tools=("tool_call" in template and "tools" in template)

    tool_format="qwen3_5" if expect_tools else ""
    config=f"""network:
  host: 127.0.0.1
  port: {args.port}
  disable_auth: true
  allowed_origins: []
  api_servers: ["OAI"]

model:
  model_dir: {model.parent}
  model_name: {model.name}
  backend: exllamav3
  max_seq_len: {args.context}
  cache_size: {args.context}
  cache_mode: FP16
  tensor_parallel: false
  gpu_split_auto: false
  gpu_split: [23, 0]
  chunk_size: 2048
  max_batch_size: {args.max_batch}
  vision: {str(expect_vision).lower()}
  reasoning: true
  reasoning_start_token: "<think>"
  reasoning_end_token: "</think>"
  tool_format: {tool_format}
  warmup: true

sampling:
  override_preset:
"""
    (tabby/"config.yml").write_text(config)

    env=os.environ.copy()
    pydev=Path(__file__).resolve().parent/".pydev/root"
    if (pydev/"usr/include/python3.12/Python.h").exists():
        inc=f"{pydev}/usr/include:{pydev}/usr/include/python3.12:{pydev}/usr/include/x86_64-linux-gnu/python3.12"
        env["CPATH"]=inc+(":"+env["CPATH"] if env.get("CPATH") else "")
        env["C_INCLUDE_PATH"]=env["CPATH"]
        env["CPLUS_INCLUDE_PATH"]=env["CPATH"]

    base=f"http://127.0.0.1:{args.port}"
    logf=open(args.log,"w",buffering=1)
    proc=subprocess.Popen([sys.executable,"main.py"],cwd=tabby,env=env,stdout=logf,stderr=subprocess.STDOUT,text=True)
    result={"passed":False,"model":str(model),"context":args.context,"max_batch":args.max_batch,"checks":{}}

    def post(payload, timeout=90, stream=False):
        return requests.post(base+"/v1/chat/completions",json=payload,timeout=timeout,stream=stream)

    try:
        ready=False
        for _ in range(90):
            if proc.poll() is not None:
                break
            try:
                rr=requests.get(base+"/v1/models",timeout=2)
                if rr.ok:
                    ready=True
                    break
            except Exception:
                pass
            time.sleep(1)
        if not ready:
            raise RuntimeError("TabbyAPI did not become ready")

        rr=requests.get(base+"/v1/models",timeout=10)
        result["checks"]["models"]={"passed":rr.ok,"status":rr.status_code}

        rr=requests.get(base+"/props",timeout=10)
        props=rr.json() if rr.ok else {}
        result["checks"]["props"]={
            "passed":rr.ok and props.get("default_generation_settings",{}).get("n_ctx")==args.context,
            "status":rr.status_code,
            "vision":props.get("modalities",{}).get("vision"),
            "n_ctx":props.get("default_generation_settings",{}).get("n_ctx"),
        }

        payload={"model":model.name,"messages":[{"role":"user","content":"Reply with one short sentence confirming the inference server is working."}],"temperature":0,"max_tokens":128,"chat_template_kwargs":{"enable_thinking":True,"reasoning_effort":"low"}}
        t=time.perf_counter()
        rr=post(payload)
        elapsed=time.perf_counter()-t
        j=rr.json()
        msg=j.get("choices",[{}])[0].get("message",{})
        assistant_text=((msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")).strip()
        result["checks"]["chat"]={
            "passed":rr.ok and bool((msg.get("content") or "").strip()),
            "status":rr.status_code,
            "elapsed_s":round(elapsed,3),
            "content":msg.get("content"),
            "reasoning_content":msg.get("reasoning_content"),
            "completion_tps":(j.get("usage") or {}).get("completion_tokens_per_sec"),
        }

        sp=dict(payload)
        sp["stream"]=True
        t=time.perf_counter()
        rr=post(sp,stream=True)
        events=0
        first=None
        for line in rr.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            if line[6:]=="[DONE]":
                break
            if first is None:
                first=time.perf_counter()-t
            events+=1
        result["checks"]["stream"]={"passed":rr.ok and events>0,"status":rr.status_code,"events":events,"first_event_s":round(first,3) if first else None}

        if expect_tools:
            tool_payload={
                "model":model.name,
                "messages":[{"role":"user","content":"Use the get_weather tool to check the weather in Toronto. Do not answer from memory."}],
                "tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
                "tool_choice":"auto","temperature":0,"max_tokens":128
            }
            rr=post(tool_payload)
            j=rr.json()
            msg=j.get("choices",[{}])[0].get("message",{})
            calls=msg.get("tool_calls") or []
            ok=bool(calls and calls[0].get("function",{}).get("name")=="get_weather" and "Toronto" in calls[0].get("function",{}).get("arguments",""))
            result["checks"]["tool_call"]={"passed":rr.ok and ok,"status":rr.status_code,"tool_calls":calls}
        else:
            result["checks"]["tool_call"]={"passed":True,"skipped":True}

        if expect_vision:
            im=Image.new("RGB",(256,256),"white")
            ImageDraw.Draw(im).rectangle((48,48,208,208),fill="red")
            b=io.BytesIO()
            im.save(b,format="PNG")
            uri="data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()
            vp={"model":model.name,"messages":[{"role":"user","content":[{"type":"text","text":"What color is the large square? Answer with one color word."},{"type":"image_url","image_url":{"url":uri}}]}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":False}}
            rr=post(vp)
            j=rr.json()
            msg=j.get("choices",[{}])[0].get("message",{})
            answer=(msg.get("content") or "").strip().lower()
            result["checks"]["vision"]={"passed":rr.ok and answer=="red","status":rr.status_code,"content":msg.get("content")}
        else:
            result["checks"]["vision"]={"passed":True,"skipped":True}

        def one(n):
            p={"model":model.name,"messages":[{"role":"user","content":f"Return only the integer {n}."}],"temperature":0,"max_tokens":8,"chat_template_kwargs":{"enable_thinking":False}}
            t=time.perf_counter()
            rr=post(p)
            elapsed=time.perf_counter()-t
            j=rr.json()
            msg=j.get("choices",[{}])[0].get("message",{})
            assistant_text=((msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")).strip()
            return {"passed":rr.ok and bool(assistant_text),"status":rr.status_code,"elapsed_s":round(elapsed,3),"content":msg.get("content"),"reasoning_content":msg.get("reasoning_content")}

        with ThreadPoolExecutor(max_workers=2) as ex:
            vals=list(ex.map(one,[17,23]))
        result["checks"]["concurrency_2"]={"passed":all(x["passed"] for x in vals),"responses":vals}

        result["passed"]=all(v.get("passed") for v in result["checks"].values())
        write_json(args.out,result)
        print(json.dumps(result,indent=2))
        raise SystemExit(0 if result["passed"] else 4)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        logf.close()

if __name__=="__main__":
    main()
