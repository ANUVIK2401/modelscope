"""Run the layer-sensitivity experiment end to end.

Loads each model family's FP16 weights, profiles every linear layer, fits the
cross-family predictor, and writes results. Weights are loaded to CPU: this
experiment needs no GPU and no eval data, which is the practical claim being
made -- the triage is cheap.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM

from analysis.paths import RESULTS_DIR, SENSITIVITY_CSV, ensure_results_dir
from models.configs import GEMMA, LLAMA
from models.env import capture_environment, set_global_seed
from research.predictor import compare_formats, fit_sensitivity_predictor
from research.sensitivity import profile_model_layers

FAMILIES: dict[str, str] = {"llama": LLAMA, "gemma": GEMMA}


def profile_family(family: str, model_id: str) -> pd.DataFrame:
    print(f"Loading {model_id} on CPU for weight profiling ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cpu"
    )
    try:
        frame = profile_model_layers(model)
    finally:
        del model
    frame["model_family"] = family
    frame["model_id"] = model_id
    return frame


def run(families: dict[str, str] = FAMILIES,
        output_csv: Path = SENSITIVITY_CSV) -> dict:
    set_global_seed()
    ensure_results_dir()

    frames = [profile_family(family, model_id) for family, model_id in families.items()]
    profile = pd.concat(frames, ignore_index=True)
    profile.to_csv(output_csv, index=False)
    print(f"Profiled {profile['layer'].nunique()} layers across "
          f"{len(families)} families -> {output_csv}")

    report: dict = {
        "environment": capture_environment(),
        "n_layers_total": int(profile["layer"].nunique()),
        "families": list(families),
        "predictors": {},
        "format_comparison": compare_formats(profile),
    }

    for method in ("fp4", "nf4", "int8"):
        try:
            report["predictors"][method] = fit_sensitivity_predictor(profile, method)
        except ValueError as exc:
            report["predictors"][method] = {"error": str(exc)}

    output_json = RESULTS_DIR / "sensitivity_report.json"
    output_json.write_text(json.dumps(report, indent=2, default=str))
    print(f"Report -> {output_json}")

    comparison = report["format_comparison"]
    print(f"\nNF4 vs FP4: mean relative advantage "
          f"{comparison['nf4_relative_advantage'] * 100:.1f}%, "
          f"NF4 better on {comparison['nf4_wins_fraction'] * 100:.0f}% of layers")
    print(f"Advantage vs tail-heaviness correlation: "
          f"{comparison['advantage_vs_tail_correlation']}")

    for method, result in report["predictors"].items():
        if "error" in result:
            continue
        best = min(result["models"].items(), key=lambda kv: kv[1]["mae_heldout"])
        print(f"{method}: best held-out predictor {best[0]}, "
              f"R2={best[1]['r2_heldout']:.3f} "
              f"(baseline R2={best[1]['r2_baseline']:.3f}), split={result['split']}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer-wise quantization sensitivity experiment."
    )
    parser.add_argument("--families", nargs="*", default=list(FAMILIES),
                        choices=list(FAMILIES))
    parser.add_argument("--output", type=Path, default=SENSITIVITY_CSV)
    args = parser.parse_args()
    run({k: FAMILIES[k] for k in args.families}, output_csv=args.output)


if __name__ == "__main__":
    main()
