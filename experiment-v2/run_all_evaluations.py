"""
MedVerify AI — Final Evaluation Suite (experiment-v2)
=====================================================
Master Evaluation Orchestration Runner

Executes the 19-step evaluation pipeline either sequentially in full or by individual step:
  --all                 Run all 16 evaluation step scripts in sequence
  --step <N>            Run only step N (e.g. --step 7 for Main Research Experiment)
  --sample <N>          Run evaluation on a limited sample size for quick validation
  --report-only         Synthesize final evaluation tables from existing outputs

Usage:
  python experiment-v2/run_all_evaluations.py --all
  python experiment-v2/run_all_evaluations.py --step 7
  python experiment-v2/run_all_evaluations.py --sample 50
"""

import os
import sys
import time
import argparse
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description="MedVerify AI Final Evaluation Runner (experiment-v2)")
    parser.add_argument("--all", action="store_true", help="Execute all evaluation steps sequentially")
    parser.add_argument("--step", type=int, default=None, help="Execute specific step (1-16)")
    parser.add_argument("--sample", type=int, default=None, help="Subset sample size for quick evaluation")
    parser.add_argument("--report-only", action="store_true", help="Synthesize final tables from existing outputs")

    args = parser.parse_args()

    print("=" * 80)
    print("MEDVERIFY AI — EXPERIMENT-V2 EVALUATION SUITE ORCHESTRATOR")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    step_map = {
        1: ("PubHealth Split & Leakage Audit", "step01_pubhealth_audit", "audit_splits"),
        2: ("BioBERT Disease Classifier Evaluation", "step02_biobert_eval", "evaluate_biobert"),
        3: ("Static FAISS Retrieval on Clinical Benchmark", "step03_retrieval_scifact_eval", "evaluate_retrieval"),
        4: ("PubMed API Live Retrieval Evaluation", "step04_pubmed_api_eval", "evaluate_pubmed_api"),
        5: ("Evidence Reliability (R_i) Hierarchy Evaluation", "step05_evidence_reliability_eval", "evaluate_evidence_reliability"),
        6: ("Claim–Evidence NLI Stance Evaluation", "step06_nli_stance_eval", "evaluate_nli"),
        7: ("Main Research Experiment (B1 vs B2 Verification)", "step07_b1_vs_b2_verification", "run_b1_vs_b2_experiment"),
        8: ("Evidence-Quality Verdict Change Ablation", "step08_verdict_change_ablation", "run_verdict_ablation"),
        9: ("Claim–Evidence Strength & Overclaim Evaluation", "step09_claim_strength_overclaim", "evaluate_overclaiming"),
        10: ("Explanation Grounding & Citation Evaluation", "step10_explanation_citation_eval", "evaluate_explanation_citations"),
        11: ("Explanation Faithfulness & BioScope Guard", "step11_faithfulness_bioscope_eval", "evaluate_faithfulness"),
        12: ("Confidence Calibration (ECE & Brier Score)", "step12_confidence_calibration", "evaluate_calibration"),
        13: ("External Real-World Claims Evaluation", "step13_external_realworld_eval", "evaluate_external_realworld"),
        14: ("CoAID Misinformation Benchmark Evaluation", "step14_coaid_evaluation", "evaluate_coaid"),
        15: ("Statistical Significance (Bootstrap CI & McNemar)", "step15_statistical_significance", "evaluate_significance"),
        16: ("Master Final Tables Synthesis (Tables 1–6)", "step16_generate_final_tables", "generate_all_tables"),
    }

    if args.report_only:
        import step16_generate_final_tables
        step16_generate_final_tables.generate_all_tables()
        return

    if args.step:
        if args.step not in step_map:
            print(f"Error: Step {args.step} not recognized. Available steps: 1 through 16.")
            sys.exit(1)
        name, module_name, func_name = step_map[args.step]
        print(f"\n[RUNNING STEP {args.step}]: {name}...")
        mod = __import__(module_name)
        func = getattr(mod, func_name)
        if args.sample and func_name in ["evaluate_biobert", "run_b1_vs_b2_experiment", "evaluate_nli", "evaluate_coaid"]:
            func(sample_size=args.sample)
        else:
            func()
        return

    # If --all or default
    steps_to_run = list(step_map.keys()) if args.all else [1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    start_time = time.time()
    for s in steps_to_run:
        name, module_name, func_name = step_map[s]
        print(f"\n>>> EXECUTING STEP {s}/16: {name}...")
        try:
            mod = __import__(module_name)
            func = getattr(mod, func_name)
            if args.sample and func_name in ["evaluate_biobert", "run_b1_vs_b2_experiment", "evaluate_nli", "evaluate_coaid"]:
                func(sample_size=args.sample)
            else:
                func()
        except Exception as e:
            print(f"Error executing step {s} ({name}): {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ALL EVALUATION STEPS COMPLETED IN {elapsed:.2f}s")
    print(f"Results stored in: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
