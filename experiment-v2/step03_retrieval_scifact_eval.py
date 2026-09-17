"""
MedVerify AI — Final Evaluation Suite (experiment-v2)
=====================================================
STEP 5: Evidence Retrieval Evaluation (Static FAISS + MiniLM against SciFact / Clinical Benchmark)

Evaluates the static FAISS dense retrieval component independently from final verification:
- Retriever: FAISS dense index + all-MiniLM-L6-v2 embeddings
- Primary Metric: nDCG@10
- Secondary Metrics: Recall@5, Recall@10, MRR
- Populates Table 2 of the final report
- Exports report to results/step03_retrieval_scifact.json
"""

import os
import sys
import json
import math
import numpy as np
from typing import List, Dict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import RESULTS_DIR, VECTOR_STORE_DIR, BACKEND_DIR
from data_loader import save_json

# Add backend directory to sys.path
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.services.retrieval_engine import FAISSRetriever

OUTPUT_FILE = os.path.join(RESULTS_DIR, "step03_retrieval_scifact.json")

# 25 Curated clinical evaluation benchmark queries across key medical domains
CLINICAL_RETRIEVAL_BENCHMARK = [
    # Diabetes
    {
        "query_id": "ret-diab-01",
        "disease_category": "Diabetes",
        "query": "Does metformin reduce cardiovascular mortality and complications in type 2 diabetes?",
        "target_keywords": ["metformin", "glycemic", "mortality", "diabetes", "cardiovascular"]
    },
    {
        "query_id": "ret-diab-02",
        "disease_category": "Diabetes",
        "query": "What is the efficacy of SGLT2 inhibitors like empagliflozin in diabetic kidney disease?",
        "target_keywords": ["sglt2", "empagliflozin", "kidney", "renal", "diabetes", "cardiovascular"]
    },
    {
        "query_id": "ret-diab-03",
        "disease_category": "Diabetes",
        "query": "Can ketogenic diet cure type 2 diabetes permanently without medication?",
        "target_keywords": ["keto", "ketogenic", "cure", "glycemic", "diet", "diabetes"]
    },
    {
        "query_id": "ret-diab-04",
        "disease_category": "Diabetes",
        "query": "How does GLP-1 receptor agonist semaglutide affect HbA1c and weight reduction in diabetes?",
        "target_keywords": ["glp-1", "semaglutide", "hba1c", "weight", "insulin", "diabetes"]
    },
    {
        "query_id": "ret-diab-05",
        "disease_category": "Diabetes",
        "query": "Risk factors and symptoms of diabetic ketoacidosis in type 1 versus type 2 diabetes",
        "target_keywords": ["ketoacidosis", "dka", "insulin", "glucose", "acidosis", "diabetes"]
    },
    # Cardiovascular Disease & Heart Attack
    {
        "query_id": "ret-cvd-01",
        "disease_category": "Cardiovascular Disease",
        "query": "Do statins reduce the risk of recurrent myocardial infarction and stroke in coronary artery disease?",
        "target_keywords": ["statin", "atorvastatin", "myocardial", "infarction", "stroke", "coronary", "cholesterol"]
    },
    {
        "query_id": "ret-cvd-02",
        "disease_category": "Heart Disease",
        "query": "What are the clinical guidelines for dual antiplatelet therapy after coronary stent placement?",
        "target_keywords": ["antiplatelet", "aspirin", "clopidogrel", "stent", "coronary", "thrombosis"]
    },
    {
        "query_id": "ret-cvd-03",
        "disease_category": "Heart Disease",
        "query": "Can vigorous coughing during a heart attack restore blood flow as a self-resuscitation measure?",
        "target_keywords": ["cough", "cpr", "heart", "attack", "resuscitation", "infarction"]
    },
    {
        "query_id": "ret-cvd-04",
        "disease_category": "Cardiovascular Disease",
        "query": "How effective is the Mediterranean diet in primary prevention of major cardiovascular events?",
        "target_keywords": ["mediterranean", "diet", "cardiovascular", "stroke", "mortality", "prevention"]
    },
    {
        "query_id": "ret-cvd-05",
        "disease_category": "Heart Disease",
        "query": "What are the primary symptoms and diagnostic biomarkers of acute ST-elevation myocardial infarction?",
        "target_keywords": ["troponin", "stemi", "myocardial", "infarction", "ecg", "chest"]
    },
    # Vaccination
    {
        "query_id": "ret-vax-01",
        "disease_category": "Vaccination",
        "query": "Is there any scientific link between the MMR vaccine and autism spectrum disorder in children?",
        "target_keywords": ["mmr", "autism", "vaccine", "measles", "wakefield", "safety"]
    },
    {
        "query_id": "ret-vax-02",
        "disease_category": "Vaccination",
        "query": "What is the clinical efficacy and safety of mRNA COVID-19 vaccines in preventing severe illness?",
        "target_keywords": ["mrna", "covid", "vaccine", "pfizer", "biontech", "moderna", "efficacy", "safety"]
    },
    {
        "query_id": "ret-vax-03",
        "disease_category": "Vaccination",
        "query": "Do COVID-19 vaccines alter human genomic DNA or cause shedding of viral particles?",
        "target_keywords": ["dna", "alter", "shedding", "mrna", "genomic", "vaccine"]
    },
    {
        "query_id": "ret-vax-04",
        "disease_category": "Vaccination",
        "query": "What are the indications and effectiveness of the recombinant shingles vaccine Shingrix in older adults?",
        "target_keywords": ["shingles", "shingrix", "zoster", "herpes", "vaccine", "efficacy"]
    },
    {
        "query_id": "ret-vax-05",
        "disease_category": "Vaccination",
        "query": "How do seasonal influenza vaccines reduce hospitalizations and mortality in elderly populations?",
        "target_keywords": ["influenza", "flu", "vaccine", "elderly", "hospitalization", "mortality"]
    },
    # General Cancer & Screening
    {
        "query_id": "ret-ca-01",
        "disease_category": "General Cancer",
        "query": "Does low-dose CT screening reduce lung cancer mortality in individuals with heavy smoking history?",
        "target_keywords": ["ct", "screening", "lung", "cancer", "mortality", "smoking"]
    },
    {
        "query_id": "ret-ca-02",
        "disease_category": "Breast Cancer",
        "query": "What are the survival benefits and recurrence risks of adjuvant endocrine therapy in ER-positive breast cancer?",
        "target_keywords": ["tamoxifen", "aromatase", "endocrine", "breast", "cancer", "recurrence"]
    },
    {
        "query_id": "ret-ca-03",
        "disease_category": "General Cancer",
        "query": "Can high-dose vitamin C infusions or alkaline diets eradicate solid malignant tumors?",
        "target_keywords": ["vitamin c", "alkaline", "cancer", "tumor", "alternative", "cure"]
    },
    # Stroke & Hypertension
    {
        "query_id": "ret-str-01",
        "disease_category": "Stroke",
        "query": "What is the effective therapeutic time window for intravenous thrombolysis with alteplase in acute ischemic stroke?",
        "target_keywords": ["alteplase", "tpa", "thrombolysis", "stroke", "ischemic", "window"]
    },
    {
        "query_id": "ret-str-02",
        "disease_category": "Stroke",
        "query": "How do direct oral anticoagulants compare to warfarin for stroke prevention in non-valvular atrial fibrillation?",
        "target_keywords": ["doac", "apixaban", "warfarin", "atrial", "fibrillation", "stroke"]
    }
]


def compute_ndcg_at_k(relevance_scores: List[float], k: int = 10) -> float:
    """Compute nDCG at rank k."""
    actual_scores = relevance_scores[:k]
    if not actual_scores:
        return 0.0

    # DCG
    dcg = sum((2 ** rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(actual_scores))

    # Ideal DCG
    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(ideal_scores))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval() -> Dict:
    print("=" * 80)
    print("EXPERIMENT-V2 — STEP 5: STATIC FAISS RETRIEVAL EVALUATION")
    print("=" * 80)
    print("Retriever Component: FAISS + all-MiniLM-L6-v2 Dense Embeddings")
    print("Primary Metric: nDCG@10")

    retriever = FAISSRetriever(VECTOR_STORE_DIR)

    recall_at_5_list = []
    recall_at_10_list = []
    mrr_list = []
    ndcg_at_10_list = []

    per_query_results = []

    for item in CLINICAL_RETRIEVAL_BENCHMARK:
        qid = item["query_id"]
        query = item["query"]
        keywords = item["target_keywords"]

        # Retrieve top 10 from FAISS
        retrieved_docs = retriever.search(query, top_k=10)

        # Compute relevance for each retrieved chunk (graded relevance 0, 1, 2)
        graded_relevances = []
        is_relevant_flags = []

        for doc in retrieved_docs:
            doc_text = (doc.get("chunk_text", "") + " " + doc.get("title", "")).lower()
            matching_kw_count = sum(1 for kw in keywords if kw.lower() in doc_text)

            if matching_kw_count >= 2:
                graded_rel = 2.0  # Highly relevant
                is_rel = True
            elif matching_kw_count == 1:
                graded_rel = 1.0  # Partially relevant
                is_rel = True
            else:
                graded_rel = 0.0  # Not relevant
                is_rel = False

            graded_relevances.append(graded_rel)
            is_relevant_flags.append(is_rel)

        # Recall@5: at least one relevant document in top 5
        r_at_5 = 1.0 if any(is_relevant_flags[:5]) else 0.0
        # Recall@10: at least one relevant document in top 10
        r_at_10 = 1.0 if any(is_relevant_flags[:10]) else 0.0

        # MRR: reciprocal rank of first relevant doc
        mrr = 0.0
        for rank, is_rel in enumerate(is_relevant_flags, start=1):
            if is_rel:
                mrr = 1.0 / rank
                break

        # nDCG@10
        ndcg_10 = compute_ndcg_at_k(graded_relevances, k=10)

        recall_at_5_list.append(r_at_5)
        recall_at_10_list.append(r_at_10)
        mrr_list.append(mrr)
        ndcg_at_10_list.append(ndcg_10)

        per_query_results.append({
            "query_id": qid,
            "query": query,
            "disease_category": item["disease_category"],
            "recall_at_5": r_at_5,
            "recall_at_10": r_at_10,
            "mrr": round(mrr, 4),
            "ndcg_at_10": round(ndcg_10, 4),
        })

    mean_r5 = float(np.mean(recall_at_5_list))
    mean_r10 = float(np.mean(recall_at_10_list))
    mean_mrr = float(np.mean(mrr_list))
    mean_ndcg10 = float(np.mean(ndcg_at_10_list))

    print("\n" + "=" * 80)
    print("TABLE 2 — EVIDENCE RETRIEVAL PERFORMANCE (STATIC FAISS + MINILM)")
    print("=" * 80)
    print(f"{'Retriever':<25} | {'Recall@5':>10} | {'Recall@10':>10} | {'MRR':>10} | {'nDCG@10 (Main)':>15}")
    print("-" * 80)
    print(f"{'FAISS + MiniLM':<25} | {mean_r5:10.4f} | {mean_r10:10.4f} | {mean_mrr:10.4f} | {mean_ndcg10:15.4f}")
    print("=" * 80)

    report = {
        "evaluation_name": "Table 2 — Evidence Retrieval Evaluation",
        "retriever": "FAISS + MiniLM",
        "num_queries": len(CLINICAL_RETRIEVAL_BENCHMARK),
        "primary_metric": "nDCG@10",
        "metrics": {
            "recall_at_5": round(mean_r5, 4),
            "recall_at_10": round(mean_r10, 4),
            "mrr": round(mean_mrr, 4),
            "ndcg_at_10": round(mean_ndcg10, 4),
        },
        "per_query_details": per_query_results,
    }

    save_json(report, OUTPUT_FILE)
    print(f"\n[OK] Retrieval evaluation report saved to: {OUTPUT_FILE}")
    return report


if __name__ == "__main__":
    evaluate_retrieval()
