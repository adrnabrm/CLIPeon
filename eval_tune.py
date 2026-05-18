#!/usr/bin/env python3
"""Tune fusion weights (clip / dino / color) via k-fold CV on labels.jsonl.

Re-ranks stored per-signal scores offline — no Chroma or GPU required.
Optimizes mean NDCG@k on validation queries each fold; reports vault holdout.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabeledResult:
    card_id: str
    relevance: int
    clip_score: float
    dino_score: float
    color_score: float


@dataclass(frozen=True)
class LabeledQuery:
    query_card_id: str
    k: int
    results: tuple[LabeledResult, ...]


@dataclass(frozen=True)
class FusionWeights:
    clip: float
    dino: float
    color: float

    def fused(self, r: LabeledResult) -> float:
        return (
            self.clip * r.clip_score
            + self.dino * r.dino_score
            + self.color * r.color_score
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "clip_weight": round(self.clip, 4),
            "dino_weight": round(self.dino, 4),
            "color_weight": round(self.color, 4),
        }

    def __str__(self) -> str:
        return f"clip={self.clip:.2f} dino={self.dino:.2f} color={self.color:.2f}"


def load_labels(path: Path) -> list[LabeledQuery]:
    queries: list[LabeledQuery] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        results = tuple(
            LabeledResult(
                card_id=r["card_id"],
                relevance=int(r["relevance"]),
                clip_score=float(r["clip_score"]),
                dino_score=float(r["dino_score"]),
                color_score=float(r["color_score"]),
            )
            for r in raw["results"]
        )
        if not results:
            raise ValueError(f"{path}:{line_no}: no rated results")
        queries.append(
            LabeledQuery(
                query_card_id=raw["query_card_id"],
                k=int(raw.get("k", 12)),
                results=results,
            )
        )
    return queries


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _dcg(relevances: list[int], k: int) -> float:
    total = 0.0
    for i, rel in enumerate(relevances[:k]):
        gain = (2**rel) - 1
        total += gain / math.log2(i + 2)
    return total


def ndcg_at_k(relevances: list[int], k: int) -> float:
    """NDCG@k with standard 2^rel - 1 gains."""
    if not relevances:
        return 0.0
    dcg = _dcg(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = _dcg(ideal, k)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def query_ndcg(query: LabeledQuery, weights: FusionWeights, ndcg_k: int) -> float:
    ranked = sorted(query.results, key=weights.fused, reverse=True)
    relevances = [r.relevance for r in ranked]
    return ndcg_at_k(relevances, ndcg_k)


def mean_ndcg(
    queries: list[LabeledQuery],
    weights: FusionWeights,
    ndcg_k: int,
) -> float:
    if not queries:
        return 0.0
    return statistics.mean(query_ndcg(q, weights, ndcg_k) for q in queries)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def weight_grid(step: float) -> list[FusionWeights]:
    """Non-negative weights on a simplex, spaced by `step` (e.g. 0.1)."""
    n = round(1.0 / step)
    out: list[FusionWeights] = []
    for i in range(n + 1):
        w_clip = i * step
        for j in range(n + 1 - i):
            w_dino = j * step
            w_color = round(1.0 - w_clip - w_dino, 10)
            if w_color < -1e-9:
                continue
            out.append(FusionWeights(w_clip, w_dino, w_color))
    return out


def best_weights_on_queries(
    queries: list[LabeledQuery],
    candidates: list[FusionWeights],
    ndcg_k: int,
) -> tuple[FusionWeights, float]:
    best_w = candidates[0]
    best_score = -1.0
    for w in candidates:
        score = mean_ndcg(queries, w, ndcg_k)
        if score > best_score:
            best_score = score
            best_w = w
    return best_w, best_score


def k_fold_indices(n: int, k: int, seed: int) -> list[list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    folds: list[list[int]] = [[] for _ in range(k)]
    for i, idx in enumerate(indices):
        folds[i % k].append(idx)
    return folds


# ---------------------------------------------------------------------------
# Main tuning pipeline
# ---------------------------------------------------------------------------


def run_tuning(
    queries: list[LabeledQuery],
    *,
    k_folds: int,
    vault_size: int,
    grid_step: float,
    ndcg_k: int,
    seed: int,
    baseline: FusionWeights,
) -> dict:
    if len(queries) < k_folds + vault_size + 1:
        raise ValueError(
            f"Need at least {k_folds + vault_size + 1} queries; got {len(queries)}"
        )

    rng = random.Random(seed)
    shuffled = list(queries)
    rng.shuffle(shuffled)

    vault = shuffled[:vault_size]
    tune = shuffled[vault_size:]

    candidates = weight_grid(grid_step)
    fold_assignment = k_fold_indices(len(tune), k_folds, seed=seed + 1)

    fold_results: list[dict] = []
    best_per_fold: list[FusionWeights] = []

    for fold_no, val_indices in enumerate(fold_assignment, 1):
        val_set = [tune[i] for i in val_indices]
        best_w, best_val_ndcg = best_weights_on_queries(val_set, candidates, ndcg_k)
        best_per_fold.append(best_w)
        fold_results.append(
            {
                "fold": fold_no,
                "val_queries": len(val_set),
                "best_weights": best_w.as_dict(),
                "val_mean_ndcg": round(best_val_ndcg, 4),
            }
        )

    # Refit: grid on all tune queries (not vault)
    final_w, tune_ndcg = best_weights_on_queries(tune, candidates, ndcg_k)

    # Average of per-fold winners (diagnostic; refit is the pick)
    avg_clip = statistics.mean(w.clip for w in best_per_fold)
    avg_dino = statistics.mean(w.dino for w in best_per_fold)
    avg_color = statistics.mean(w.color for w in best_per_fold)
    fold_avg_w = FusionWeights(avg_clip, avg_dino, avg_color)

    cv_scores = []
    for fold_no, val_indices in enumerate(fold_assignment, 1):
        val_set = [tune[i] for i in val_indices]
        cv_scores.append(mean_ndcg(val_set, final_w, ndcg_k))

    baseline_tune = mean_ndcg(tune, baseline, ndcg_k)
    baseline_vault = mean_ndcg(vault, baseline, ndcg_k)
    baseline_all = mean_ndcg(queries, baseline, ndcg_k)

    vault_ndcg = mean_ndcg(vault, final_w, ndcg_k)
    all_ndcg = mean_ndcg(queries, final_w, ndcg_k)

    vault_ids = [q.query_card_id for q in vault]

    return {
        "n_queries": len(queries),
        "n_tune": len(tune),
        "n_vault": len(vault),
        "vault_query_ids": vault_ids,
        "k_folds": k_folds,
        "grid_step": grid_step,
        "ndcg_k": ndcg_k,
        "seed": seed,
        "n_candidates": len(candidates),
        "baseline_weights": baseline.as_dict(),
        "baseline_mean_ndcg": {
            "tune": round(baseline_tune, 4),
            "vault": round(baseline_vault, 4),
            "all": round(baseline_all, 4),
        },
        "fold_avg_weights": fold_avg_w.as_dict(),
        "per_fold": fold_results,
        "final_weights": final_w.as_dict(),
        "final_mean_ndcg": {
            "tune": round(tune_ndcg, 4),
            "vault": round(vault_ndcg, 4),
            "all": round(all_ndcg, 4),
            "cv_on_tune_folds_mean": round(statistics.mean(cv_scores), 4),
            "cv_on_tune_folds_std": round(
                statistics.stdev(cv_scores) if len(cv_scores) > 1 else 0.0, 4
            ),
        },
        "delta_vs_baseline": {
            "tune": round(tune_ndcg - baseline_tune, 4),
            "vault": round(vault_ndcg - baseline_vault, 4),
            "all": round(all_ndcg - baseline_all, 4),
        },
    }


def _print_report(report: dict) -> None:
    bw = report["baseline_weights"]
    fw = report["final_weights"]
    bm = report["baseline_mean_ndcg"]
    fm = report["final_mean_ndcg"]
    d = report["delta_vs_baseline"]

    print(f"Labels: {report['n_queries']} queries "
          f"({report['n_tune']} tune, {report['n_vault']} vault holdout)")
    print(f"Grid: {report['n_candidates']} candidates, step={report['grid_step']}, "
          f"{report['k_folds']}-fold, NDCG@{report['ndcg_k']}, seed={report['seed']}")
    print()
    print("Baseline weights "
          f"(clip={bw['clip_weight']}, dino={bw['dino_weight']}, color={bw['color_weight']})")
    print(f"  tune  NDCG = {bm['tune']:.4f}")
    print(f"  vault NDCG = {bm['vault']:.4f}")
    print(f"  all   NDCG = {bm['all']:.4f}")
    print()
    for fold in report["per_fold"]:
        w = fold["best_weights"]
        print(
            f"Fold {fold['fold']}: best val NDCG={fold['val_mean_ndcg']:.4f}  "
            f"(clip={w['clip_weight']}, dino={w['dino_weight']}, color={w['color_weight']})"
        )
    print()
    print("Final weights (refit on all tune queries)")
    print(f"  clip={fw['clip_weight']}, dino={fw['dino_weight']}, color={fw['color_weight']}")
    print(f"  tune  NDCG = {fm['tune']:.4f}  (delta {d['tune']:+.4f})")
    print(f"  vault NDCG = {fm['vault']:.4f}  (delta {d['vault']:+.4f})")
    print(f"  all   NDCG = {fm['all']:.4f}  (delta {d['all']:+.4f})")
    print(f"  CV mean±std on tune folds: {fm['cv_on_tune_folds_mean']:.4f} "
          f"± {fm['cv_on_tune_folds_std']:.4f}")
    print()
    print("Vault query ids (held out during tuning):")
    print("  " + ", ".join(report["vault_query_ids"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="K-fold tune clip/dino/color fusion weights on eval labels.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "eval" / "labels.jsonl",
    )
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--vault-size", type=int, default=10)
    parser.add_argument("--grid-step", type=float, default=0.1)
    parser.add_argument("--ndcg-k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baseline-clip",
        type=float,
        default=None,
        help="Baseline clip weight (default: from first label record index_params)",
    )
    parser.add_argument("--baseline-dino", type=float, default=None)
    parser.add_argument("--baseline-color", type=float, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON report (default: data/eval/tune_report.json)",
    )
    args = parser.parse_args()

    labels_path = args.labels_path.expanduser().resolve()
    queries = load_labels(labels_path)

    if args.baseline_clip is None:
        first = json.loads(labels_path.read_text(encoding="utf-8").splitlines()[0])
        ip = first.get("index_params", {})
        baseline = FusionWeights(
            float(ip.get("clip_weight", 0.4)),
            float(ip.get("dino_weight", 0.4)),
            float(ip.get("color_weight", 0.2)),
        )
    else:
        baseline = FusionWeights(
            args.baseline_clip,
            args.baseline_dino if args.baseline_dino is not None else 0.4,
            args.baseline_color if args.baseline_color is not None else 0.2,
        )

    report = run_tuning(
        queries,
        k_folds=args.k_folds,
        vault_size=args.vault_size,
        grid_step=args.grid_step,
        ndcg_k=args.ndcg_k,
        seed=args.seed,
        baseline=baseline,
    )
    report["labels_path"] = str(labels_path)

    out_path = args.output
    if out_path is None:
        out_path = labels_path.parent / "tune_report.json"
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(out_path)

    _print_report(report)
    print()
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
