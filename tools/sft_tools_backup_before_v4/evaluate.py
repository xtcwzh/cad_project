#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate.py — 基线评测脚本

对比: 预测(jsonl: part + pred) vs 真值(samples jsonl 的 assistant 内容)

指标:
  json_valid_rate      预测能否 json.loads
  schema_ok_rate       v2 还原器的结构门禁（不代表 Creo 重建成功）
  feat_count_mae       特征数平均绝对误差
  type_seq_similarity  类型序列相似度 (difflib ratio, 0~1)
  type_f1_micro        特征类型多重集合 F1 (micro)
  parent_edge_f1       父子边(位置归一) F1
  dim_rel_err_median   对齐特征尺寸相对误差中位数 (gt!=0)
  dim_exact_rate       尺寸完全一致率 (rel err < 1e-6)

用法:
  python evaluate.py --tag mock --split val
  (默认读项目 data/sft_v2，可通过 --sft-root 指定)
"""
import argparse
import difflib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from artifact_io import restore_label
from label_codec import encode as encode_v2

ROOT = Path(r"D:\pilot_dataset")


def norm_type(t):
    return str(t or "").upper()


def to_positions(feats):
    """特征数组 -> (类型序列, id->位置映射, 位置归一的父子边集合)"""
    types = [norm_type(f.get("type")) for f in feats]
    id2pos = {f.get("id"): i for i, f in enumerate(feats) if f.get("id") is not None}
    edges = set()
    for i, f in enumerate(feats):
        for p in f.get("par", []) or []:
            j = id2pos.get(p)
            if j is not None:
                edges.add((i, j))
    return types, id2pos, edges


def multiset_f1(gt_types, pred_types):
    gc, pc = Counter(gt_types), Counter(pred_types)
    matched = sum(min(gc[t], pc.get(t, 0)) for t in gc)
    p = matched / sum(pc.values()) if pc else 0.0
    r = matched / sum(gc.values()) if gc else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


def edge_prf(gt_edges, pred_edges):
    if not gt_edges and not pred_edges:
        return 1.0, 1.0, 1.0
    inter = len(gt_edges & pred_edges)
    p = inter / len(pred_edges) if pred_edges else 0.0
    r = inter / len(gt_edges) if gt_edges else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


def eval_one(gt_label, pred_text, context=None, registry=None):
    m = {"json_valid": False, "schema_ok": False}
    try:
        pred = json.loads(pred_text)
        m["json_valid"] = True
    except Exception:
        m["error"] = "json parse failed"
        return m
    try:
        restored = restore_label(pred, context, registry)
        if pred.get('label_version') != gt_label.get('label_version'):
            raise ValueError('Prediction label version differs from target')
        if pred.get('label_version') == 3:
            pred = encode_v2(restored, context.get('name', ''))
            gt_label = encode_v2(restore_label(gt_label, context, registry), context.get('name', ''))
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        m['error'] = f'Invalid v2 label: {exc}'
        return m
    feats = pred["feats"]
    m["schema_ok"] = True

    gt_types, gt_id2pos, gt_edges = to_positions(gt_label["feats"])
    pr_types, pr_id2pos, pr_edges = to_positions(feats)

    m["feat_count_gt"] = len(gt_label["feats"])
    m["feat_count_pred"] = len(feats)
    m["feat_count_mae"] = abs(len(feats) - len(gt_label["feats"]))

    sm = difflib.SequenceMatcher(None, gt_types, pr_types)
    m["type_seq_similarity"] = round(sm.ratio(), 4)

    p, r_, f1 = multiset_f1(gt_types, pr_types)
    m["type_precision"] = round(p, 4)
    m["type_recall"] = round(r_, 4)
    m["type_f1_micro"] = round(f1, 4)

    ep, er, ef1 = edge_prf(gt_edges, pr_edges)
    m["parent_edge_f1"] = round(ef1, 4)

    # 尺寸: 类型序列对齐块内按位置配对
    rels = []
    exact = 0
    total = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        for k in range(i2 - i1):
            gf = gt_label["feats"][i1 + k]
            pf = feats[j1 + k]
            gd = gf.get("dims") or []
            pd = pf.get("dims") or []
            for a, b in zip(gd, pd):
                try:
                    a, b = float(a['value']), float(b['value'])
                except (TypeError, ValueError):
                    continue
                total += 1
                if abs(a) > 1e-9:
                    e = abs(b - a) / abs(a)
                    rels.append(e)
                    if e < 1e-6:
                        exact += 1
    m["dim_pairs_compared"] = total
    if rels:
        m["dim_rel_err_median"] = round(statistics.median(rels), 6)
        m["dim_rel_err_mean"] = round(statistics.mean(rels), 6)
        m["dim_exact_rate"] = round(exact / len(rels), 4)
    else:
        m["dim_rel_err_median"] = None
        m["dim_exact_rate"] = None
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument('--sft-root', type=Path,
                    default=Path(__file__).resolve().parents[2] / 'data' / 'sft_v3')
    args = ap.parse_args()
    if Path(args.tag).name != args.tag or args.tag in ('.', '..'):
        raise ValueError('tag must be a directory name')

    pred_path = args.sft_root / "predictions" / args.tag / f"{args.split}.pred.jsonl"
    samples_path = args.sft_root / "samples" / f"{args.split}.jsonl"

    gt_map = {}
    contexts = {}
    schema_path = args.sft_root / 'xml_schemas.json'
    registry = json.loads(schema_path.read_text(encoding='utf-8')) if schema_path.exists() else None
    for line in samples_path.read_text(encoding="utf-8").strip().split("\n"):
        rec = json.loads(line)
        gt_map[rec["part"]] = json.loads(rec["messages"][2]["content"])
        contexts[rec['part']] = (json.loads(rec['messages'][1]['content'])['provided_context']
                                if gt_map[rec['part']].get('label_version') == 3 else None)
        restore_label(gt_map[rec['part']], contexts[rec['part']], registry)

    per_part = []
    modes = set()
    seen = set()
    for line in pred_path.read_text(encoding="utf-8").strip().split("\n"):
        rec = json.loads(line)
        part = rec["part"]
        if part in seen:
            raise ValueError(f'Duplicate prediction for {part}')
        seen.add(part)
        modes.add(rec.get('mode', 'unknown'))
        gt = gt_map.get(part)
        if gt is None:
            per_part.append({"part": part, "error": "no ground truth"})
            continue
        m = eval_one(gt, rec["pred"], contexts[part], registry)
        m["part"] = part
        per_part.append(m)

    # 汇总
    n = len(per_part)
    agg = {
        "tag": args.tag, "split": args.split, "n": n,
        'modes': sorted(modes),
        'expected_samples': len(gt_map),
        'prediction_coverage': len(seen & gt_map.keys()) / len(gt_map) if gt_map else 0,
        "json_valid_rate": round(sum(1 for x in per_part if x.get("json_valid")) / n, 4) if n else 0,
        "schema_ok_rate": round(sum(1 for x in per_part if x.get("schema_ok")) / n, 4) if n else 0,
    }
    ok = [x for x in per_part if x.get("schema_ok")]
    for key in ("feat_count_mae", "type_seq_similarity", "type_f1_micro",
                "parent_edge_f1", "dim_rel_err_median", "dim_exact_rate"):
        vals = [x[key] for x in ok if x.get(key) is not None]
        agg[key] = round(statistics.mean(vals), 4) if vals else None

    rep = {"aggregate": agg, "per_part": per_part}
    out = args.sft_root / "predictions" / args.tag / f"{args.split}.eval_report.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[eval] tag={args.tag} split={args.split} n={n}")
    for k, v in agg.items():
        if k not in ("tag", "split", "n"):
            print(f"  {k:24s} {v}")
    print(f"  报告: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
