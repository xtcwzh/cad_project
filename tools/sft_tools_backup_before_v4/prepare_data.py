#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_data.py — 保真 v2 数据准备（先验证转换，非最终紧凑训练格式）

输入: D:\\pilot_dataset\\{feature_trees, evidence, parts, brep}
输出: D:\\cad_project\\data\\sft_v2\\（可通过 --out 指定）
    clean\\<name>.json        清洁数据对（剥 base64 的完整树 + 质量真值 + 路径索引）
    simplified\\<name>.label.json   保真标签（保留 XML 和几何映射）
    samples\\{train,val,test}.jsonl   SFT 样本（messages 格式，按零件划分）
    report.json               统计报告

用法: python prepare_data.py [--root D:\\pilot_dataset] [--out <新输出目录>]
"""
import argparse
import json
import random
import sys
from pathlib import Path

from label_codec import encode, sketch_codec


def strip_snapshot(tree):
    from copy import deepcopy
    t = deepcopy(tree)
    t.pop("native_snapshot", None)
    return t


def compact_sketch(sk):
    return sketch_codec(sk)


def simplify_tree(tree, part_name):
    return encode(tree, part_name)


SYSTEM_PROMPT = (
    "Recover a Creo parametric feature tree from BREP evidence. Output JSON only, "
    "using label_version=2, part, and feats. Root aliases: features->feats, "
    "material_name->material, relations->rel. Feature aliases: feat_id->id, "
    "feat_number->n, feat_type_name->type, feat_type->type_id, feat_name->name, "
    "feat_status->st, feat_subtype->sub, parent_ids->par, child_ids->chld, "
    "dimensions->dims, sketch->sk, elem_refs->refs. Sketch aliases: "
    "location_matrix->m, intent_manager->im, entities->e, dimensions->d, "
    "projection_refs->pr, constraints->c. Entity aliases: ent_type->t, "
    "json_ent_id->id. All other Creo export fields keep their original names. "
    "Dimensions are full objects with dim_id, symbol, value and type. "
    "Keep XML, constraints, geometry mappings and references self-contained. "
    "Order features by n. Do not include native_snapshot. "
    "Evidence face IDs are not original Creo geometry IDs. "
    "Material and density are not determined by geometry alone."
)


def build_sample(label, evidence):
    # Include all available evidence, without repeating the lossy brief.
    # No labels, XML or other ground-truth history may enter the user message.
    payload = {k: v for k, v in evidence.items() if k != "llm_brief"}
    user = "Recover the Creo feature tree from this BREP evidence JSON:\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"))
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": json.dumps(label, ensure_ascii=False, separators=(",", ":"))},
    ]}


# ---------------------------------------------------------------- 主流程

def difficulty(label):
    n = len(label["feats"])
    return 0 if n <= 10 else (1 if n <= 25 else 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\pilot_dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "sft_v2")
    args = ap.parse_args()
    root = Path(args.root)

    sft = args.out
    clean_dir = sft / "clean"
    simp_dir = sft / "simplified"
    samp_dir = sft / "samples"
    for d in (clean_dir, simp_dir, samp_dir):
        d.mkdir(parents=True, exist_ok=True)

    parts = sorted((root / "parts").glob("*.prt"))
    if not parts:
        raise ValueError("No source parts found")
    stats = {"ok": 0, "fail": [], "sizes": [], "label_sizes": [],
             "sample_sizes": []}
    records = []

    for prt in parts:
        name = prt.stem
        ft_path = root / "feature_trees" / f"{name}_feature_tree.json"
        ev_path = root / "evidence" / f"{name}.evidence.v1.json"
        stp_path = root / "brep" / f"{name}.stp"
        if not ft_path.exists():
            stats["fail"].append(f"{name}: no feature tree")
            continue
        if not ev_path.exists():
            stats["fail"].append(f"{name}: no evidence")
            continue

        try:
            tree = json.loads(ft_path.read_text(encoding="utf-8"))
            ev = json.loads(ev_path.read_text(encoding="utf-8"))
        except Exception as e:
            stats["fail"].append(f"{name}: parse error {e}")
            continue

        # 1) 清洁对: 剥 base64 + 质量真值(来自 evidence 的 pythonOCC 计算)
        density = float(tree.get("density") or 0)
        g = ev.get("global", {})
        volume = g.get("volume")
        mass_truth = None
        if volume is not None:
            mass_truth = {
                "volume_mm3": volume,
                "bbox_min": g.get("bbox_min"),
                "bbox_max": g.get("bbox_max"),
                "face_count": g.get("face_count"),
                "density": density,
                "mass_g": round(volume * density, 6) if density else None,
                "source": "pythonOCC from STEP",
            }
        clean = {
            "part": name,
            "paths": {
                "prt": str(prt),
                "step": str(stp_path),
                "evidence": str(ev_path),
                "feature_tree_raw": str(ft_path),
            },
            "mass_truth": mass_truth,
            "label_tree": strip_snapshot(tree),
        }
        (clean_dir / f"{name}.json").write_text(
            json.dumps(clean, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")

        # 2) 简化标签
        label = simplify_tree(tree, name)
        (simp_dir / f"{name}.label.json").write_text(
            json.dumps(label, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")

        # 3) SFT 样本
        sample = build_sample(label, ev)

        records.append({
            "name": name, "label": label, "sample": sample,
            "raw_kb": ft_path.stat().st_size / 1024,
            "clean_kb": (clean_dir / f"{name}.json").stat().st_size / 1024,
            "label_kb": (simp_dir / f"{name}.label.json").stat().st_size / 1024,
            "user_chars": len(sample["messages"][1]["content"]),
            "asst_chars": len(sample["messages"][2]["content"]),
            "diff": difficulty(label),
        })
        stats["ok"] += 1

    # 4) 划分: 按难度分层, 齿轮/弹簧强制进 test(泛化组)
    random.seed(args.seed)
    test_kw = ("齿轮", "压簧", "拉簧")
    groups = {0: [], 1: [], 2: []}
    for r in records:
        if any(k in r["name"] for k in test_kw):
            r["split"] = "test"
        else:
            groups[r["diff"]].append(r)
    for d, rs in groups.items():
        random.shuffle(rs)
        n = len(rs)
        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.10))
        for i, r in enumerate(rs):
            r.setdefault("split", "test" if i < n_test else
                         ("val" if i < n_test + n_val else "train"))

    splits = {"train": [], "val": [], "test": []}
    for r in records:
        splits[r["split"]].append(r)

    for split, rs in splits.items():
        with open(samp_dir / f"{split}.jsonl", "w", encoding="utf-8") as fp:
            for r in rs:
                # 顶层加 part 字段便于评测对账; 正式训练时可一行脚本剔除
                fp.write(json.dumps({"part": r["name"], **r["sample"]},
                                    ensure_ascii=False) + "\n")

    # 5) 报告
    rep = {
        "label_version": 2,
        "source_root": str(root),
        "output_root": str(sft),
        "creo_rebuild_verified": False,
        "training_ready": False,
        "readiness_note": "Lossless reference format; requires token budgets and Creo rebuild validation",
        "ok": stats["ok"],
        "fail": stats["fail"],
        "counts": {k: len(v) for k, v in splits.items()},
        "raw_total_kb": round(sum(r["raw_kb"] for r in records), 1),
        "clean_total_kb": round(sum(r["clean_kb"] for r in records), 1),
        "label_total_kb": round(sum(r["label_kb"] for r in records), 1),
        "avg": {
            "raw_kb": round(sum(r["raw_kb"] for r in records) / max(1, len(records)), 1),
            "label_kb": round(sum(r["label_kb"] for r in records) / max(1, len(records)), 1),
            "user_chars": round(sum(r["user_chars"] for r in records) / max(1, len(records))),
            "asst_chars": round(sum(r["asst_chars"] for r in records) / max(1, len(records))),
        },
        "parts": [{"name": r["name"], "split": r["split"],
                   "feats": len(r["label"]["feats"]),
                   "label_kb": round(r["label_kb"], 1),
                   "user_chars": r["user_chars"], "assistant_chars": r["asst_chars"]} for r in records],
    }
    (sft / "report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[prepare] ok={rep['ok']} fail={len(stats['fail'])} "
          f"train/val/test={rep['counts']}")
    print(f"[prepare] raw {rep['raw_total_kb']}KB -> clean {rep['clean_total_kb']}KB "
          f"-> label {rep['label_total_kb']}KB")
    print(f"[prepare] avg sample: user {rep['avg']['user_chars']} chars, "
          f"assistant {rep['avg']['asst_chars']} chars")
    if stats["fail"]:
        for f in stats["fail"]:
            print("  FAIL:", f)
    return 0 if not stats["fail"] else 1


if __name__ == "__main__":
    sys.exit(main())
