import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.evaluation.dataset import load_dataset, EvalSample
from src.evaluation.metrics import evaluate_prd
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalResult:
    variant_name: str
    sample_count: int
    avg_completeness: float
    avg_feasibility: float
    avg_consistency: float
    avg_relevance: float
    avg_overall: float
    per_sample: list[dict] = field(default_factory=list)
    raw_results: list[dict] = field(default_factory=list)


def run_evaluation(
    variant_name: str,
    generate_fn,
    dataset: list[EvalSample],
    model: str | None = None,
) -> EvalResult:
    results = []
    for i, sample in enumerate(dataset):
        logger.info("Evaluating [%s] sample %d/%d: %s", variant_name, i + 1, len(dataset), sample.product_idea[:50])
        try:
            prd_md = generate_fn(sample.product_idea)
            scores = evaluate_prd(prd_md, sample.product_idea, model=model)
            results.append({
                "sample_index": i,
                "product_idea": sample.product_idea,
                "category": sample.category,
                "scores": scores,
            })
        except Exception as e:
            logger.error("Sample %d failed: %s", i, e)
            results.append({
                "sample_index": i,
                "product_idea": sample.product_idea,
                "category": sample.category,
                "scores": {"completeness": 0, "feasibility": 0, "consistency": 0, "relevance": 0},
                "error": str(e),
            })

    n = len(results)
    avg_completeness = sum(r["scores"]["completeness"] for r in results) / n if n else 0
    avg_feasibility = sum(r["scores"]["feasibility"] for r in results) / n if n else 0
    avg_consistency = sum(r["scores"]["consistency"] for r in results) / n if n else 0
    avg_relevance = sum(r["scores"]["relevance"] for r in results) / n if n else 0
    avg_overall = (avg_completeness + avg_feasibility + avg_consistency + avg_relevance) / 4

    return EvalResult(
        variant_name=variant_name,
        sample_count=n,
        avg_completeness=round(avg_completeness, 1),
        avg_feasibility=round(avg_feasibility, 1),
        avg_consistency=round(avg_consistency, 1),
        avg_relevance=round(avg_relevance, 1),
        avg_overall=round(avg_overall, 1),
        per_sample=[{
            "product_idea": r["product_idea"],
            "scores": r["scores"],
        } for r in results],
        raw_results=results,
    )


def run_ablation_experiment(
    variants: dict,
    dataset: list[EvalSample],
    model: str | None = None,
) -> list[EvalResult]:
    experiment_results = []
    for variant_name, generate_fn in variants.items():
        logger.info("=== Running variant: %s ===", variant_name)
        result = run_evaluation(variant_name, generate_fn, dataset, model=model)
        experiment_results.append(result)
        logger.info("Variant %s: avg_overall=%.1f", variant_name, result.avg_overall)
    return experiment_results


def generate_comparison_table(results: list[EvalResult]) -> str:
    header = "| 版本 | 完整性 | 可行性 | 一致性 | 相关性 | 综合均分 |"
    sep = "|------|--------|--------|--------|--------|----------|"
    rows = [f"| {r.variant_name} | {r.avg_completeness} | {r.avg_feasibility} | {r.avg_consistency} | {r.avg_relevance} | {r.avg_overall} |" for r in results]
    return "\n".join([header, sep] + rows)


def run_from_cli(
    dataset_path: str | None = None,
    output_path: str | None = None,
    ablation: bool = False,
):
    """CLI entry point for evaluation."""
    from src.workflow.graph import run_workflow
    from src.output.json_to_markdown import convert_to_prd_markdown

    dataset = load_dataset(dataset_path)
    logger.info("Loaded %d evaluation samples", len(dataset))

    def generate_basic(product_idea: str) -> str:
        result = run_workflow(product_idea=product_idea)
        final_json = result.get("final_prd_json", {})
        return convert_to_prd_markdown(final_json)

    def generate_full(product_idea: str) -> str:
        from src.workflow.multi_agent.graph import run_multi_agent_workflow
        result = run_multi_agent_workflow(product_idea=product_idea)
        final_json = result.get("final_prd_json", {})
        return convert_to_prd_markdown(final_json)

    if ablation:
        variants = {
            "V1_baseline": generate_basic,
            "V2_multi_agent": generate_full,
        }
        results = run_ablation_experiment(variants, dataset)
        table = generate_comparison_table(results)
        print("\n消融实验对比结果：\n")
        print(table)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("# PRD Generator 消融实验报告\n\n")
                f.write(table)
                f.write("\n\n## 各样本详情\n\n")
                for r in results:
                    f.write(f"### {r.variant_name}\n")
                    for s in r.per_sample:
                        f.write(f"- {s['product_idea'][:60]}...: {s['scores']}\n")
                    f.write("\n")
            print(f"\n报告已保存至: {output_path}")
    else:
        result = run_evaluation("default", generate_full, dataset)
        print(f"\n评估完成 — 综合均分: {result.avg_overall}/10")
        print(f"  完整性: {result.avg_completeness}")
        print(f"  可行性: {result.avg_feasibility}")
        print(f"  一致性: {result.avg_consistency}")
        print(f"  相关性: {result.avg_relevance}")

    return results if ablation else result


if __name__ == "__main__":
    import sys

    ablation_flag = "--ablation" in sys.argv
    dataset_path = None
    output_path = None

    for i, arg in enumerate(sys.argv):
        if arg == "--dataset" and i + 1 < len(sys.argv):
            dataset_path = sys.argv[i + 1]
        if arg == "--output" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]

    run_from_cli(dataset_path=dataset_path, output_path=output_path, ablation=ablation_flag)
