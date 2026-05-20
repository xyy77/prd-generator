"""Ablation experiment on a 3-sample subset for faster results."""
import json, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Initialize logging so logger messages don't get dropped
from src.utils.logger import setup_logging
setup_logging()

from src.evaluation.dataset import load_dataset
from src.evaluation.metrics import evaluate_prd
from src.workflow.graph import run_workflow
from src.workflow.multi_agent.graph import run_multi_agent_workflow
from src.output.json_to_markdown import convert_to_prd_markdown

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# Load dataset, pick 1 sample per category
all_samples = load_dataset()
samples = []
seen = set()
for s in all_samples:
    if s.category not in seen:
        samples.append(s)
        seen.add(s.category)
    if len(samples) >= 3:
        break

log(f"Loaded {len(all_samples)} samples, using {len(samples)} (1 per category)")
for s in samples:
    log(f"  [{s.category}] {s.product_idea[:60]}...")

variants = {
    "V1_baseline": lambda idea: convert_to_prd_markdown(
        run_workflow(product_idea=idea).get("final_prd_json", {})),
    "V2_multi_agent": lambda idea: convert_to_prd_markdown(
        run_multi_agent_workflow(product_idea=idea).get("final_prd_json", {})),
}

results = {}
for vname, gen_fn in variants.items():
    log(f"{'='*50}")
    log(f"Starting variant: {vname}")
    log(f"{'='*50}")
    variant_scores = []
    for i, sample in enumerate(samples):
        log(f"[{vname}] Sample {i+1}/{len(samples)}: {sample.product_idea[:60]}")
        t0 = time.time()
        try:
            prd_md = gen_fn(sample.product_idea)
            gen_time = time.time() - t0
            log(f"  Generated in {gen_time:.0f}s, {len(prd_md)} chars")
            t1 = time.time()
            scores = evaluate_prd(prd_md, sample.product_idea)
            eval_time = time.time() - t1
            overall = round(sum(scores.values())/4, 1)
            log(f"  Scores: {scores} | Overall={overall} | Eval in {eval_time:.0f}s")
            variant_scores.append({
                "product_idea": sample.product_idea,
                "category": sample.category,
                "gen_time_s": round(gen_time, 0),
                "eval_time_s": round(eval_time, 0),
                "scores": scores,
            })
        except Exception as e:
            log(f"  FAILED: {e}")
            variant_scores.append({
                "product_idea": sample.product_idea,
                "category": sample.category,
                "error": str(e),
                "scores": {"completeness": 0, "feasibility": 0, "consistency": 0, "relevance": 0},
            })
    results[vname] = variant_scores
    log(f"Completed {vname}")

# Averages
def avg(lst, key):
    vals = [s["scores"][key] for s in lst]
    return round(sum(vals)/len(vals), 1) if vals else 0

log("")
log("=" * 60)
log("ABLATION RESULTS")
log("=" * 60)

print("\n| 版本 | 完整性 | 可行性 | 一致性 | 相关性 | 综合均分 | 平均生成耗时 |")
print("|------|--------|--------|--------|--------|----------|-------------|")
for vname, scores in results.items():
    a = [avg(scores, k) for k in ["completeness","feasibility","consistency","relevance"]]
    overall = round(sum(a)/4, 1)
    avg_gen = round(sum(s.get("gen_time_s",0) for s in scores)/len(scores), 0)
    print(f"| {vname} | {a[0]} | {a[1]} | {a[2]} | {a[3]} | {overall} | {avg_gen:.0f}s |")

print("\n## Per-sample details\n")
for vname, scores in results.items():
    print(f"### {vname}")
    for s in scores:
        sc = s["scores"]
        overall = round(sum(sc.values())/4, 1)
        print(f"- [{s['category']}] {s['product_idea'][:80]}... → {overall} ({sc})")
    print()

# Save
output_path = Path("results/ablation.md")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    f.write("# PRD Generator 消融实验报告\n\n")
    f.write(f"实验时间: {time.strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"样本数: {len(samples)}\n\n")
    f.write("## 对比结果\n\n")
    f.write("| 版本 | 完整性 | 可行性 | 一致性 | 相关性 | 综合均分 |\n")
    f.write("|------|--------|--------|--------|--------|----------|\n")
    for vname, scores in results.items():
        a = [avg(scores, k) for k in ["completeness","feasibility","consistency","relevance"]]
        overall = round(sum(a)/4, 1)
        f.write(f"| {vname} | {a[0]} | {a[1]} | {a[2]} | {a[3]} | {overall} |\n")
    f.write("\n## 各样本详情\n\n")
    for vname, scores in results.items():
        f.write(f"### {vname}\n")
        for s in scores:
            sc = s["scores"]
            overall = round(sum(sc.values())/4, 1)
            f.write(f"- [{s['category']}] {s['product_idea'][:80]}... → 综合={overall} ({sc})\n")
        f.write("\n")

log(f"Report saved to {output_path}")
log("DONE")
