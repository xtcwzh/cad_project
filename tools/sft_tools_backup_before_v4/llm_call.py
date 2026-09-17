#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
llm_call.py — 基线调用脚本（OpenAI 兼容 API / mock 自测模式）

真实 API 用法（环境变量三件套，任选一家）:
    阿里百炼: LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
              LLM_MODEL=qwen-plus (或 qwen-max / qwen2.5-72b-instruct)
    智谱:     LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
              LLM_MODEL=glm-4-flash (免费) / glm-4-plus
    本地vLLM: LLM_BASE_URL=http://<服务器>:8000/v1  LLM_MODEL=<模型名>
    （LLM_API_KEY 三种情况都要设，本地 vLLM 随便填非空）
    然后:
    python llm_call.py --split val --tag qwen-plus [--limit 5] [--json-mode]

链路自测（不需要任何 Key）:
    python llm_call.py --split val --tag mock --mock
    （用真值标签降级伪造预测: 随机删特征 + 扰动尺寸, 验证评测全链路）

输出: D:\\cad_project\\data\\sft_v3\\predictions\\<tag>\\<split>.pred.jsonl
      每行: {"part": ..., "pred": "<模型原始输出文本>"}

真实调用另需 --tokenizer <本地模型目录> --context-length <服务实际上限>，
并设置 --max-tokens。v2 标签明显变长，预算不足会拒绝运行，不静默截断。
"""
import argparse
import copy
import json
import os
import random
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(r"D:\pilot_dataset")
DEFAULT_SFT = Path(__file__).resolve().parents[2] / 'data' / 'sft_v3'


def inference_messages(messages):
    """Dataset is a single-turn system/user/target record, not chat history."""
    if [m.get('role') for m in messages] != ['system', 'user', 'assistant']:
        raise ValueError('Expected system/user/assistant training sample')
    return copy.deepcopy(messages[:2])


def chat(base_url, api_key, model, messages, temperature=0.0,
         max_tokens=8192, json_mode=False, timeout=300):
    if any(m.get('role') == 'assistant' for m in messages):
        raise ValueError('Refusing assistant target in baseline request')
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + api_key}
    body = {"model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers=headers, method="POST")
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choice = data["choices"][0]
            if choice.get('finish_reason') == 'length':
                raise ValueError('Output truncated: increase --max-tokens; do not score as complete')
            content = choice["message"]["content"]
            if content and content.strip():
                return content
            last_err = RuntimeError("empty content")
        except Exception as e:
            last_err = e
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"API failed after 3 retries: {last_err}")


def mock_predict(gt_label, rng):
    """真值标签降级伪造预测: 删 1~2 个特征 + 一半尺寸扰动±8%"""
    pred = copy.deepcopy(gt_label)
    feats = pred.get("feats", [])
    if len(feats) > 2 and rng.random() < 0.7:
        for _ in range(rng.randint(1, 2)):
            if len(feats) > 1:
                feats.pop(rng.randrange(len(feats)))
    for f in feats:
        from semantic_v3 import expand, compact
        data = expand(f['data']) if pred.get('label_version') == 3 else f
        if data.get("dims"):
            for dim in data['dims']:
                if rng.random() < 0.5:
                    dim['value'] *= 1 + rng.uniform(-0.08, 0.08)
        if pred.get('label_version') == 3:
            f['data'] = compact(data)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--tag", required=True, help="输出子目录名(模型标识)")
    ap.add_argument("--limit", type=int, default=0, help="只跑前N条, 0=全部")
    ap.add_argument("--mock", action="store_true", help="自测模式, 无需API")
    ap.add_argument("--json-mode", action="store_true",
                    help="请求 response_format=json_object")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument('--sft-root', type=Path, default=DEFAULT_SFT)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--tokenizer', type=Path, help='Local deployed tokenizer directory; required for real calls')
    ap.add_argument('--context-length', type=int, help='Actual server context limit')
    args = ap.parse_args()

    samples_path = args.sft_root / "samples" / f"{args.split}.jsonl"
    lines = samples_path.read_text(encoding="utf-8").strip().split("\n")
    if args.limit:
        lines = lines[:args.limit]

    if Path(args.tag).name != args.tag or args.tag in ('.', '..'):
        raise ValueError('tag must be a directory name')
    out_dir = args.sft_root / "predictions" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}.pred.jsonl"

    base_url = os.environ.get("LLM_BASE_URL", "")
    api_key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "")
    if not args.mock and not all((base_url, api_key, model)):
        raise ValueError('Set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL')
    from artifact_io import restore_label
    schema_path = args.sft_root / 'xml_schemas.json'
    registry = json.loads(schema_path.read_text(encoding='utf-8')) if schema_path.exists() else None
    tokenizer = None
    if not args.mock:
        if not args.tokenizer or not args.context_length:
            raise ValueError('Real v2 calls require --tokenizer and --context-length for budget validation')
        from check_lengths import load_tokenizer, token_budget
        tokenizer = load_tokenizer(args.tokenizer)
    # Validate all selected targets BEFORE creating a prediction file.
    for line in lines:
        sample = json.loads(line)
        inference_messages(sample['messages'])
        target = json.loads(sample['messages'][2]['content'])
        context = (json.loads(sample['messages'][1]['content'])['provided_context']
                   if target.get('label_version') == 3 else None)
        restore_label(target, context, registry)
        if tokenizer:
            budget = token_budget(sample['messages'], tokenizer, args.context_length, args.max_tokens)
            if not budget['fits']:
                raise ValueError(f"Sample {sample['part']} exceeds token budget: {budget}; no truncation allowed")

    rng = random.Random(0)
    n_ok = 0
    with open(out_path, "w" if args.overwrite else "x", encoding="utf-8") as fp:
        for i, line in enumerate(lines):
            rec = json.loads(line)
            part = rec.get("part", f"sample_{i}")
            messages = rec["messages"]
            if args.mock:
                gt = json.loads(messages[2]["content"])
                pred_text = json.dumps(
                    mock_predict(gt, rng), ensure_ascii=False,
                    separators=(",", ":"))
            else:
                pred_text = chat(base_url, api_key, model, inference_messages(messages),
                                 max_tokens=args.max_tokens,
                                 json_mode=args.json_mode)
            fp.write(json.dumps({"part": part, "pred": pred_text,
                                'mode': 'mock' if args.mock else 'model',
                                'model': None if args.mock else model},
                                ensure_ascii=False) + "\n")
            fp.flush()
            n_ok += 1
            print(f"[{i+1}/{len(lines)}] {part} "
                  f"({len(pred_text)} chars)")

    print(f"\n[llm_call] done: {n_ok}/{len(lines)} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
