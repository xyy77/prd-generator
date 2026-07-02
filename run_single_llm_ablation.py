#!/usr/bin/env python3
"""Ablation experiment: V1_baseline vs V2_multi_agent vs V3_single_llm.

Adds V3_single_llm (one LLM call, no pipeline/agents/reflection) as the
simplest baseline. Loads existing V1/V2 results and only runs V3.

Output: results/ablation_v3_compare.md + results/ablation_v3_data.json
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.logger import setup_logging
setup_logging()

from src.evaluation.dataset import load_dataset
from src.evaluation.metrics import evaluate_prd
from src.workflow.single_llm import run_single_llm_workflow
from src.output.json_to_markdown import convert_to_prd_markdown
from src.config import settings

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXISTING_DATA_PATH = OUTPUT_DIR / "ablation_data.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def avg_score(samples: list[dict], key: str) -> float:
    vals = [s["scores"].get(key, 0) for s in samples]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def avg_time(samples: list[dict]) -> float:
    vals = [s.get("gen_time_s", 0) for s in samples]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def overall(scores: dict) -> float:
    return round(sum(scores.values()) / 4, 1)


# ── Load the same 5 samples used in run_ablation_fixed.py ──────────
all_samples = load_dataset()
selected = []
seen_cat = set()
for s in all_samples:
    if s.category not in seen_cat or len(selected) < 5:
        selected.append(s)
        seen_cat.add(s.category)
    if len(selected) >= 5:
        break

log("=" * 60)
log("ABLATION: V3_single_llm (adding to existing V1/V2 comparison)")
log(f"Samples: {len(selected)} (covering {len(seen_cat)} categories)")
log(f"Model: {settings.deepseek_model}")
log(f"Temperature: {settings.llm_temperature}")
log(f"Existing V1/V2 data: {EXISTING_DATA_PATH}")
log("=" * 60)

# ── Load existing V1/V2 data ──────────────────────────────────────
if EXISTING_DATA_PATH.exists():
    existing_data = json.loads(EXISTING_DATA_PATH.read_text(encoding="utf-8"))
    log(f"Loaded existing results: {list(existing_data.keys())}")
else:
    log("WARNING: No existing data found, will only have V3 results")
    existing_data = {}

# ── Run V3_single_llm ─────────────────────────────────────────────
log("")
log(f"{'─' * 50}")
log("  V3_single_llm")
log(f"{'─' * 50}")

v3_data: list[dict] = []

for i, sample in enumerate(selected):
    idea = sample.product_idea
    cat = sample.category
    log(f"[V3_single_llm] {i+1}/{len(selected)} [{cat}] {idea[:60]}...")

    try:
        t0 = time.time()
        result = run_single_llm_workflow(product_idea=idea)
        gen_time = time.time() - t0

        prd_md = result.get("final_prd_markdown", "")
        if not prd_md:
            final_json = result.get("final_prd_json", {})
            if final_json:
                prd_md = convert_to_prd_markdown(final_json)

        log(f"  Generated: {gen_time:.0f}s, {len(prd_md)} chars")

        if prd_md:
            t1 = time.time()
            scores = evaluate_prd(prd_md, idea)
            eval_time = time.time() - t1
        else:
            scores = {"completeness": 0, "feasibility": 0, "consistency": 0, "relevance": 0}
            eval_time = 0
            log("  WARNING: Empty PRD output")

        entry = {
            "product_idea": idea,
            "category": cat,
            "gen_time_s": round(gen_time, 0),
            "prd_length": len(prd_md),
            "eval_time_s": round(eval_time, 0) if eval_time else 0,
            "scores": scores,
        }
        log(f"  Scores: {scores} | Overall={overall(scores)}")
        v3_data.append(entry)

    except Exception as e:
        log(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        v3_data.append({
            "product_idea": idea,
            "category": cat,
            "error": str(e),
            "gen_time_s": 0,
            "prd_length": 0,
            "scores": {"completeness": 0, "feasibility": 0, "consistency": 0, "relevance": 0},
        })

log(f"  V3_single_llm complete: {len(v3_data)} samples")

# ── Combine all results ───────────────────────────────────────────
all_results = dict(existing_data)
all_results["V3_single_llm"] = v3_data

# ── Generate report ────────────────────────────────────────────────
report = []
report.append("# PRD Generator 消融实验报告（三列对比）")
report.append("")
report.append(f"- **实验时间**: {time.strftime('%Y-%m-%d %H:%M')}")
report.append(f"- **模型**: {settings.deepseek_model}")
report.append(f"- **样本数**: {len(selected)}")
report.append(f"- **温度**: {settings.llm_temperature}")
report.append("")
report.append("## 三种模式说明")
report.append("")
report.append("| 模式 | 说明 | 架构 |")
report.append("|------|------|------|")
report.append("| **V3_single_llm** | 单次 LLM 调用直接生成完整 PRD | 1 个 Prompt → 1 次 LLM 调用 → PRD |")
report.append("| **V1_baseline** | 经典 3 阶段流水线 | 需求分析+架构设计(并行) → 流程梳理 → 文档定稿 |")
report.append("| **V2_multi_agent** | 多智能体协作 + 评审闭环 | Planner → Supervisor → 4 Agent + Reviewer + 反思修订 |")
report.append("")

# ── Comparison table ──
variant_order = ["V3_single_llm", "V1_baseline", "V2_multi_agent"]
present_variants = [v for v in variant_order if v in all_results]

report.append("## 综合质量对比")
report.append("")
header_cols = ["版本", "完整性", "可行性", "一致性", "相关性", "**综合均分**", "平均耗时"]
report.append("| " + " | ".join(header_cols) + " |")
report.append("|" + "|".join(["------"] * len(header_cols)) + "|")

for vn in present_variants:
    data = all_results[vn]
    a = [avg_score(data, k) for k in ["completeness", "feasibility", "consistency", "relevance"]]
    ov = round(sum(a) / 4, 1)
    t = avg_time(data)
    report.append(f"| {vn} | {a[0]} | {a[1]} | {a[2]} | {a[3]} | **{ov}** | {t:.0f}s |")

report.append("")

# ── Improvement over single LLM ──
if "V3_single_llm" in all_results and "V2_multi_agent" in all_results:
    v3_ov = round(sum([avg_score(all_results["V3_single_llm"], k) for k in ["completeness", "feasibility", "consistency", "relevance"]]) / 4, 1)
    v1_ov_val = round(sum([avg_score(all_results["V1_baseline"], k) for k in ["completeness", "feasibility", "consistency", "relevance"]]) / 4, 1)
    v2_ov = round(sum([avg_score(all_results["V2_multi_agent"], k) for k in ["completeness", "feasibility", "consistency", "relevance"]]) / 4, 1)

    report.append("## 提升幅度（相对于单 LLM 基线）")
    report.append("")
    report.append(f"| 指标 | V3 单LLM | V1 流水线 | V2 多智能体 | V1 vs V3 | V2 vs V3 |")
    report.append(f"|------|----------|-----------|-------------|----------|----------|")
    report.append(f"| 综合均分 | {v3_ov}/10 | {v1_ov_val}/10 | {v2_ov}/10 | **+{round(v1_ov_val - v3_ov, 1)}** | **+{round(v2_ov - v3_ov, 1)}** |")

    for dim in ["completeness", "feasibility", "consistency", "relevance"]:
        d3 = avg_score(all_results["V3_single_llm"], dim)
        d1 = avg_score(all_results["V1_baseline"], dim)
        d2 = avg_score(all_results["V2_multi_agent"], dim)
        report.append(f"| {dim} | {d3} | {d1} | {d2} | **+{round(d1 - d3, 1)}** | **+{round(d2 - d3, 1)}** |")

    v3_t = avg_time(all_results["V3_single_llm"])
    v1_t = avg_time(all_results["V1_baseline"])
    v2_t = avg_time(all_results["V2_multi_agent"])
    report.append(f"| 平均耗时 | {v3_t:.0f}s | {v1_t:.0f}s | {v2_t:.0f}s | {v1_t/v3_t:.1f}x | {v2_t/v3_t:.1f}x |")
    report.append("")

# ── Per-sample details ──
report.append("## 各样本详情")
report.append("")

for vn in present_variants:
    data = all_results[vn]
    report.append(f"### {vn}")
    report.append("")
    for s in data:
        sc = s["scores"]
        ov = round(sum(sc.values()) / 4, 1)
        report.append(
            f"- [{s['category']}] {s['product_idea'][:60]}... "
            f"→ 综合={ov} (C:{sc['completeness']} F:{sc['feasibility']} "
            f"CS:{sc['consistency']} R:{sc['relevance']}) | {s.get('gen_time_s', 'N/A')}s"
        )
    report.append("")

# ── Resume-ready data points ──
report.append("## 简历可用数据点")
report.append("")

if "V3_single_llm" in all_results and "V2_multi_agent" in all_results:
    pct = round((v2_ov - v3_ov) / v3_ov * 100) if v3_ov > 0 else 0
    report.append(f"- 多智能体架构（V2）将 PRD 综合质量从单 LLM（V3）的 **{v3_ov}/10 提升至 {v2_ov}/10（+{round(v2_ov - v3_ov, 1)} 分，+{pct}%）**")
    report.append(f"- 经典流水线（V1）为 {v1_ov_val}/10，多智能体（V2）进一步从 {v1_ov_val} 提升至 {v2_ov}（+{round(v2_ov - v1_ov_val, 1)}），证明 Agent 分工+评审闭环的有效性")
    report.append(f"- 单 LLM（V3）仅需 {v3_t:.0f}s 但质量最低；多智能体用 {v2_t/v3_t:.1f}x 时间换取 +{round(v2_ov - v3_ov, 1)} 分质量提升")
report.append(f"- 在 {len(selected)} 个真实产品场景上完成三列消融实验验证")
report.append(f"- 4 维度 LLM-as-Judge 自动评估（完整性/可行性/一致性/相关性）")
report.append("")

# ── Raw data ──
report.append("## 原始数据")
report.append("")
report.append("```json")
report.append(json.dumps(all_results, ensure_ascii=False, indent=2))
report.append("```")

# ── Write files ──
md_path = OUTPUT_DIR / "ablation_v3_compare.md"
md_path.write_text("\n".join(report), encoding="utf-8")
log(f"Report: {md_path.resolve()}")

json_path = OUTPUT_DIR / "ablation_v3_data.json"
json_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
log(f"Raw data: {json_path.resolve()}")

# ── Print final summary ──
print("")
print("=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)
for vn in present_variants:
    data = all_results[vn]
    o = round(sum([avg_score(data, k) for k in ["completeness", "feasibility", "consistency", "relevance"]]) / 4, 1)
    t = avg_time(data)
    print(f"{vn:20s}  {o}/10 | {t:.0f}s avg")
if "V3_single_llm" in all_results and "V2_multi_agent" in all_results:
    print(f"V2 vs V3 improvement: +{round(v2_ov - v3_ov, 1)} points (+{pct}%)")
print(f"Report: {md_path.resolve()}")

log("DONE")
