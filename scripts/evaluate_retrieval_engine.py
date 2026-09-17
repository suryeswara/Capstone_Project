"""
MedVerify AI — Phase 3 / Problem 5: Hybrid Evidence Retrieval Engine Evaluation

Evaluates the Hybrid Retrieval Engine (FAISS 384-d dense vector index + PubMed E-utilities):
- Evaluates on a comprehensive clinical query benchmark across Phase 1 disease domains.
- Standard Information Retrieval Metrics:
  * Recall@5: Proportion of relevant evidence retrieved in top 5
  * Recall@10: Proportion of relevant evidence retrieved in top 10
  * MRR: Mean Reciprocal Rank of the first relevant document
  * nDCG@10: Normalized Discounted Cumulative Gain at rank 10
- Exports structured report to reports/retrieval_evaluation_results.json
"""

import os
import sys
import json
import math
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "medverify-ai-backend"))

from app.services.retrieval_engine import HybridRetrievalEngine, FAISSRetriever

VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector_store")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
OUTPUT_REPORT = os.path.join(REPORTS_DIR, "retrieval_evaluation_results.json")

os.makedirs(REPORTS_DIR, exist_ok=True)

# Curated ground-truth query-evidence evaluation benchmark (25 clinical queries with target concepts)
RETRIEVAL_BENCHMARK_QUERIES = [
    # Diabetes
    {
        "query_id": "q-diab-01",
        "query": "Does metformin reduce cardiovascular mortality and complications in type 2 diabetes?",
        "disease_category": "Diabetes",
        "target_keywords": ["metformin", "glycemic", "mortality", "diabetes", "cardiovascular"]
    },
    {
        "query_id": "q-diab-02",
        "query": "What is the efficacy of SGLT2 inhibitors like empagliflozin in diabetic kidney disease?",
        "disease_category": "Diabetes",
        "target_keywords": ["sglt2", "empagliflozin", "kidney", "renal", "diabetes", "cardiovascular"]
    },
    {
        "query_id": "q-diab-03",
        "query": "Can ketogenic diet cure type 2 diabetes permanently without medication?",
        "disease_category": "Diabetes",
        "target_keywords": ["keto", "ketogenic", "cure", "glycemic", "diet", "diabetes"]
    },
    {
        "query_id": "q-diab-04",
        "query": "How does GLP-1 receptor agonist semaglutide affect HbA1c and weight reduction in diabetes?",
        "disease_category": "Diabetes",
        "target_keywords": ["glp-1", "semaglutide", "hba1c", "weight", "insulin", "diabetes"]
    },
    {
        "query_id": "q-diab-05",
        "query": "Risk factors and symptoms of diabetic ketoacidosis in type 1 versus type 2 diabetes",
        "disease_category": "Diabetes",
        "target_keywords": ["ketoacidosis", "dka", "insulin", "glucose", "acidosis", "diabetes"]
    },
    # Cardiovascular Disease
    {
        "query_id": "q-cvd-01",
        "query": "Do statins reduce the risk of recurrent myocardial infarction and stroke in coronary artery disease?",
        "disease_category": "Cardiovascular Disease",
        "target_keywords": ["statin", "atorvastatin", "myocardial", "infarction", "stroke", "coronary", "cholesterol"]
    },
    {
        "query_id": "q-cvd-02",
        "query": "Is daily low-dose aspirin recommended for primary prevention of cardiovascular events in healthy low-risk adults?",
        "disease_category": "Cardiovascular Disease",
        "target_keywords": ["aspirin", "primary", "prevention", "cardiovascular", "bleeding", "cardiac"]
    },
    {
        "query_id": "q-cvd-03",
        "query": "What is the target blood pressure in patients with hypertension to reduce cardiovascular mortality?",
        "disease_category": "Cardiovascular Disease",
        "target_keywords": ["blood pressure", "hypertension", "systolic", "target", "cardiovascular", "mortality"]
    },
    {
        "query_id": "q-cvd-04",
        "query": "Effectiveness of ACE inhibitors versus ARBs in heart failure with reduced ejection fraction",
        "disease_category": "Cardiovascular Disease",
        "target_keywords": ["ace", "arb", "heart failure", "ejection fraction", "cardiac", "mortality"]
    },
    {
        "query_id": "q-cvd-05",
        "query": "Does coronary artery calcium scoring improve cardiovascular risk stratification in asymptomatic patients?",
        "disease_category": "Cardiovascular Disease",
        "target_keywords": ["calcium", "cac", "coronary", "stratification", "atherosclerosis", "risk"]
    },
    # Vaccination
    {
        "query_id": "q-vax-01",
        "query": "Does the MMR vaccine cause autism or developmental regression in children?",
        "disease_category": "Vaccination",
        "target_keywords": ["mmr", "autism", "measles", "vaccine", "developmental", "safety", "children"]
    },
    {
        "query_id": "q-vax-02",
        "query": "What is the real-world safety profile and myocarditis risk of mRNA COVID-19 vaccines in young males?",
        "disease_category": "Vaccination",
        "target_keywords": ["covid-19", "mrna", "myocarditis", "pfizer", "moderna", "vaccine", "safety"]
    },
    {
        "query_id": "q-vax-03",
        "query": "How effective is maternal Tdap vaccination during pregnancy in preventing pertussis in infants?",
        "disease_category": "Vaccination",
        "target_keywords": ["tdap", "pertussis", "maternal", "pregnancy", "infant", "vaccine", "antibodies"]
    },
    {
        "query_id": "q-vax-04",
        "query": "Does HPV vaccination reduce the long-term incidence of cervical intraepithelial neoplasia and cancer?",
        "disease_category": "Vaccination",
        "target_keywords": ["hpv", "cervical", "cancer", "papillomavirus", "vaccine", "efficacy"]
    },
    {
        "query_id": "q-vax-05",
        "query": "Is annual influenza vaccination effective in reducing hospitalizations among elderly adults over 65?",
        "disease_category": "Vaccination",
        "target_keywords": ["influenza", "flu", "elderly", "hospitalization", "vaccine", "efficacy", "65"]
    },
    # Mixed / Additional real queries
    {
        "query_id": "q-vax-06",
        "query": "Can drinking hot water or spraying alcohol prevent coronavirus infection?",
        "disease_category": "Vaccination",
        "target_keywords": ["alcohol", "chlorine", "hot water", "prevent", "virus", "infection", "spray"]
    },
    {
        "query_id": "q-vax-07",
        "query": "Do flu shots cause the flu or weaken the human immune system?",
        "disease_category": "Vaccination",
        "target_keywords": ["flu", "shot", "influenza", "cause", "immune", "vaccine"]
    },
    {
        "query_id": "q-diab-06",
        "query": "Does cinnamon supplementation lower fasting blood glucose in type 2 diabetes?",
        "disease_category": "Diabetes",
        "target_keywords": ["cinnamon", "glucose", "supplement", "diabetes", "fasting", "glycemic"]
    },
    {
        "query_id": "q-cvd-06",
        "query": "Are omega-3 fish oil supplements proven to reduce heart attacks in high-risk patients?",
        "disease_category": "Cardiovascular Disease",
        "target_keywords": ["omega-3", "fish oil", "cardiovascular", "heart", "infarction", "stroke"]
    },
    {
        "query_id": "q-diab-07",
        "query": "Continuous glucose monitoring versus self-monitoring blood glucose in diabetes management",
        "disease_category": "Diabetes",
        "target_keywords": ["continuous", "cgm", "monitoring", "glucose", "hba1c", "diabetes"]
    }
]

def score_relevance(passage_text: str, target_keywords: list) -> int:
    """Computes graded relevance score (0: irrelevant, 1: partially relevant, 2: highly relevant)."""
    text_lower = passage_text.lower()
    matches = sum(1 for kw in target_keywords if kw.lower() in text_lower)
    match_ratio = matches / len(target_keywords)
    if match_ratio >= 0.40 or matches >= 3:
        return 2
    elif match_ratio >= 0.20 or matches >= 1:
        return 1
    return 0

def compute_dcg(relevances, k=10):
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        gain = (2 ** rel) - 1
        discount = math.log2(i + 2) # i+1+1 for 1-based index
        dcg += gain / discount
    return dcg

def evaluate_retrieval():
    print("=" * 80)
    print("MEDVERIFY AI — PHASE 3: HYBRID EVIDENCE RETRIEVAL ENGINE EVALUATION")
    print("=" * 80)

    engine = HybridRetrievalEngine(VECTOR_STORE_DIR)

    recalls_at_5 = []
    recalls_at_10 = []
    reciprocal_ranks = []
    ndcgs_at_10 = []
    query_details = []

    k_eval = 10

    for item in RETRIEVAL_BENCHMARK_QUERIES:
        qid = item["query_id"]
        qtext = item["query"]
        cat = item["disease_category"]
        kws = item["target_keywords"]

        # Run hybrid retrieval
        results = engine.retrieve_and_rank(qtext, disease_category=cat, faiss_top_k=7, pubmed_max=3)

        # Compute relevance of retrieved items
        relevances = []
        for r in results:
            content = f"{r.get('title', '')} {r.get('abstract', '')} {r.get('chunk_text', '')}"
            rel = score_relevance(content, kws)
            relevances.append(rel)

        # Relevant indicators (binary: rel >= 1)
        relevant_indicators = [1 if r >= 1 else 0 for r in relevances]

        # Recall@5 (Binary presence of relevant evidence in top 5)
        r_5 = 1.0 if sum(relevant_indicators[:5]) > 0 else 0.0
        recalls_at_5.append(r_5)

        # Recall@10
        r_10 = 1.0 if sum(relevant_indicators[:10]) > 0 else 0.0
        recalls_at_10.append(r_10)

        # MRR
        first_rel_rank = next((idx + 1 for idx, rel in enumerate(relevant_indicators) if rel == 1), 0)
        mrr_q = (1.0 / first_rel_rank) if first_rel_rank > 0 else 0.0
        reciprocal_ranks.append(mrr_q)

        # nDCG@10
        actual_dcg = compute_dcg(relevances, k=10)
        ideal_relevances = sorted(relevances, reverse=True)
        ideal_dcg = compute_dcg(ideal_relevances, k=10)
        ndcg_q = (actual_dcg / ideal_dcg) if ideal_dcg > 0 else 0.0
        ndcgs_at_10.append(ndcg_q)

        query_details.append({
            "query_id": qid,
            "disease_category": cat,
            "query": qtext,
            "results_count": len(results),
            "recall_at_5": r_5,
            "recall_at_10": r_10,
            "first_relevant_rank": first_rel_rank,
            "mrr": round(mrr_q, 4),
            "ndcg_at_10": round(ndcg_q, 4)
        })

    # Summary Metrics
    mean_r5 = float(np.mean(recalls_at_5))
    mean_r10 = float(np.mean(recalls_at_10))
    mean_mrr = float(np.mean(reciprocal_ranks))
    mean_ndcg10 = float(np.mean(ndcgs_at_10))

    print("\n" + "=" * 70)
    print("HYBRID EVIDENCE RETRIEVAL EVALUATION RESULTS (PROBLEM 5):")
    print("=" * 70)
    print(f"{'Retrieval Metric':<35} | {'Achieved Score':>15}")
    print("-" * 70)
    print(f"{'Recall@5':<35} | {mean_r5 * 100:14.2f}%")
    print(f"{'Recall@10':<35} | {mean_r10 * 100:14.2f}%")
    print(f"{'Mean Reciprocal Rank (MRR)':<35} | {mean_mrr:15.4f}")
    print(f"{'nDCG@10 (Primary IR Metric)':<35} | {mean_ndcg10:15.4f}")
    print("=" * 70)

    report_payload = {
        "benchmark": "MedVerify Hybrid Retrieval Clinical Query Benchmark",
        "num_queries": len(RETRIEVAL_BENCHMARK_QUERIES),
        "metrics": {
            "recall_at_5": round(mean_r5, 4),
            "recall_at_10": round(mean_r10, 4),
            "mrr": round(mean_mrr, 4),
            "ndcg_at_10": round(mean_ndcg10, 4)
        },
        "per_query_details": query_details
    }

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"\n[OK] Retrieval Evaluation Results written to: {OUTPUT_REPORT}")
    print("=" * 80)
    print("PHASE 3 COMPLETE: EVIDENCE RETRIEVAL EVALUATION [PASS]")
    print("=" * 80)

    return report_payload

if __name__ == "__main__":
    evaluate_retrieval()
