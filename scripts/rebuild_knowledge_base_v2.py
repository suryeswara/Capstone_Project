"""
MedVerify AI — Rebuild Comprehensive FAISS Knowledge Base
==========================================================

Builds a production-quality FAISS evidence index from TWO sources:

  Source 1: PubHealth main_text + explanation fields
            - 8,770 records with rich fact-check explanations
            - Covers all 22 disease categories
            - Uses fact-check explanations as evidence passages
            - Uses main_text excerpts (first 800 chars) as secondary evidence

  Source 2: NCBI PubMed real biomedical abstracts
            - Fetched via PubMed E-utilities API
            - 3 queries per disease category × 22 categories = 66 queries
            - Up to 8 abstracts per query
            - Real peer-reviewed clinical evidence

Evidence chunks are stored as:
  - chunk_text: The actual evidence passage (NOT the claim itself)
  - source_tier: WHO Guideline / Systematic Review / RCT / PubMed Article etc.
  - pub_year: Actual publication year for recency decay
  - disease_category: Mapped to one of 22 canonical categories
  - verdict_stance: Supporting / Contradicted / Neutral

This replaces the old 387-vector claim-only index with a diverse,
category-balanced evidence corpus suitable for real NLI stance detection.
"""

import os
import re
import json
import time
import hashlib
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "med_datasets", "pubhealth_medical.csv.xls")
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector_store")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "vector_metadata.json")

# Back up old index
BACKUP_DIR = os.path.join(VECTOR_STORE_DIR, "backup_v1")
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# 22 Canonical Disease Categories
DISEASE_CATEGORIES = [
    "COVID-19", "General Cancer", "Influenza",
    "Reproductive Health / Abortion", "Breast Cancer", "HIV/AIDS",
    "Ebola Virus", "Alzheimers Disease", "Prostate Cancer",
    "Diabetes", "Obesity & Weight Management", "Opioids & Pain Management",
    "Measles / MMR", "Depression", "Heart Disease",
    "Smoking & Tobacco", "Pregnancy & Maternal Health", "Heart Attack",
    "Stroke", "Lung Cancer", "Autism Spectrum Disorder", "Vaccination",
]

TOPIC_ALIAS_MAP = {
    "covid-19": "COVID-19", "covid": "COVID-19", "coronavirus": "COVID-19",
    "aids": "HIV/AIDS", "hiv": "HIV/AIDS",
    "flu": "Influenza", "influenza": "Influenza",
    "heart attack": "Heart Attack", "myocardial infarction": "Heart Attack",
    "heart disease": "Heart Disease", "coronary artery disease": "Heart Disease",
    "breast cancer": "Breast Cancer", "prostate cancer": "Prostate Cancer",
    "lung cancer": "Lung Cancer", "cancer": "General Cancer",
    "alzheimer": "Alzheimers Disease", "alzheimer's": "Alzheimers Disease",
    "diabetes": "Diabetes", "measles": "Measles / MMR", "mmr": "Measles / MMR",
    "ebola": "Ebola Virus", "opioid": "Opioids & Pain Management",
    "obesity": "Obesity & Weight Management", "depression": "Depression",
    "smoking": "Smoking & Tobacco", "tobacco": "Smoking & Tobacco",
    "stroke": "Stroke", "pregnancy": "Pregnancy & Maternal Health",
    "maternal": "Pregnancy & Maternal Health",
    "abortion": "Reproductive Health / Abortion",
    "autism": "Autism Spectrum Disorder",
    "vaccination": "Vaccination", "vaccine": "Vaccination",
}

# Verdict mapping from PubHealth labels
VERDICT_MAP = {
    "true": "Supported", "True": "Supported",
    "false": "Contradicted", "False": "Contradicted",
    "mixture": "Mixed", "Mixture": "Mixed",
    "unproven": "Insufficient Evidence", "Unproven": "Insufficient Evidence",
}

# Source tier inference from explanation text patterns
def infer_source_tier(explanation: str, sources: str = "") -> str:
    text = (explanation + " " + sources).lower()
    if any(k in text for k in ["systematic review", "meta-analysis", "cochrane"]):
        return "Systematic Review / Meta-Analysis"
    elif any(k in text for k in ["randomized", "rct", "clinical trial", "double-blind"]):
        return "Randomized Controlled Trial (RCT)"
    elif any(k in text for k in ["who ", "world health", "cdc ", "centers for disease"]):
        return "WHO Guideline"
    elif any(k in text for k in ["cohort", "observational", "case-control", "prospective"]):
        return "Cohort Study / Observational"
    elif any(k in text for k in ["pubmed", "journal", "published", "study"]):
        return "PubMed Article"
    elif any(k in text for k in ["preprint", "biorxiv", "medrxiv"]):
        return "Preprint (bioRxiv/medRxiv)"
    else:
        return "PubMed Article"


def canonicalize_topic(topic_str):
    import pandas as pd
    if pd.isna(topic_str):
        return None
    key = str(topic_str).strip().lower()
    return TOPIC_ALIAS_MAP.get(key, None)


# PubMed search queries for each disease category (for categories with insufficient PubHealth coverage)
PUBMED_QUERIES = {
    "COVID-19": [
        "COVID-19 treatment efficacy systematic review",
        "SARS-CoV-2 vaccine effectiveness hospitalization prevention",
        "COVID-19 long covid symptoms management clinical trial",
    ],
    "General Cancer": [
        "cancer screening effectiveness systematic review",
        "chemotherapy versus immunotherapy outcomes meta-analysis",
        "cancer prevention lifestyle factors cohort study",
    ],
    "Influenza": [
        "influenza vaccine efficacy prevention systematic review",
        "oseltamivir influenza treatment clinical trial",
        "influenza pandemic preparedness public health guidelines",
    ],
    "Reproductive Health / Abortion": [
        "reproductive health contraception efficacy review",
        "abortion safety complications systematic review",
        "fertility treatment IVF success rates meta-analysis",
    ],
    "Breast Cancer": [
        "breast cancer screening mammography effectiveness meta-analysis",
        "breast cancer treatment tamoxifen herceptin clinical trial",
        "breast cancer risk factors prevention systematic review",
    ],
    "HIV/AIDS": [
        "HIV antiretroviral therapy treatment outcomes meta-analysis",
        "PrEP HIV prevention efficacy systematic review",
        "HIV vaccine development clinical trial results",
    ],
    "Ebola Virus": [
        "Ebola virus disease treatment supportive care review",
        "Ebola vaccine efficacy rVSV-ZEBOV clinical trial",
        "Ebola outbreak containment public health response",
    ],
    "Alzheimers Disease": [
        "Alzheimer disease treatment cholinesterase inhibitors meta-analysis",
        "dementia prevention cognitive decline risk factors review",
        "amyloid beta immunotherapy Alzheimer clinical trial",
    ],
    "Prostate Cancer": [
        "prostate cancer screening PSA test systematic review",
        "prostate cancer treatment radical prostatectomy outcomes",
        "prostate cancer active surveillance versus treatment trial",
    ],
    "Diabetes": [
        "type 2 diabetes metformin glycemic control meta-analysis",
        "insulin therapy diabetes complications prevention trial",
        "SGLT2 inhibitors cardiovascular outcomes diabetes review",
    ],
    "Obesity & Weight Management": [
        "obesity weight loss intervention systematic review",
        "bariatric surgery outcomes long term follow up meta-analysis",
        "GLP-1 receptor agonist weight management clinical trial",
    ],
    "Opioids & Pain Management": [
        "opioid addiction treatment medication assisted review",
        "chronic pain management non-opioid alternatives meta-analysis",
        "opioid epidemic prevention harm reduction strategies",
    ],
    "Measles / MMR": [
        "MMR vaccine safety autism systematic review",
        "measles vaccination coverage herd immunity review",
        "measles outbreak prevention public health guidelines",
    ],
    "Depression": [
        "depression treatment SSRI efficacy meta-analysis",
        "cognitive behavioral therapy depression systematic review",
        "exercise mental health depression clinical trial",
    ],
    "Heart Disease": [
        "coronary heart disease statin treatment meta-analysis",
        "heart failure management ACE inhibitor beta blocker review",
        "cardiovascular disease prevention lifestyle modification trial",
    ],
    "Smoking & Tobacco": [
        "smoking cessation intervention effectiveness meta-analysis",
        "tobacco harm reduction e-cigarette systematic review",
        "secondhand smoke health effects cardiovascular disease",
    ],
    "Pregnancy & Maternal Health": [
        "maternal health prenatal care outcomes systematic review",
        "gestational diabetes management treatment guidelines",
        "preeclampsia prevention aspirin low dose clinical trial",
    ],
    "Heart Attack": [
        "myocardial infarction acute treatment reperfusion meta-analysis",
        "heart attack prevention aspirin statin review",
        "cardiac rehabilitation post MI outcomes systematic review",
    ],
    "Stroke": [
        "stroke treatment thrombolysis mechanical thrombectomy trial",
        "stroke prevention anticoagulation atrial fibrillation review",
        "stroke rehabilitation recovery outcomes meta-analysis",
    ],
    "Lung Cancer": [
        "lung cancer screening low-dose CT systematic review",
        "non-small cell lung cancer immunotherapy clinical trial",
        "lung cancer smoking cessation risk reduction meta-analysis",
    ],
    "Autism Spectrum Disorder": [
        "autism spectrum disorder intervention behavioral therapy review",
        "ASD early diagnosis screening effectiveness meta-analysis",
        "autism causes genetics environmental factors systematic review",
    ],
    "Vaccination": [
        "childhood vaccination safety adverse effects meta-analysis",
        "vaccine hesitancy public health misinformation review",
        "herd immunity vaccination threshold population study",
    ],
}


def fetch_pubmed_abstracts(query: str, max_results: int = 8) -> list:
    """Fetch structured abstracts from NCBI PubMed API."""
    encoded_query = urllib.parse.quote(query)
    search_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&term={encoded_query}&retmode=json&retmax={max_results}"
    )
    req = urllib.request.Request(search_url, headers={"User-Agent": "MedVerifyAI/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            id_list = data.get("esearchresult", {}).get("idlist", [])

        if not id_list:
            return []

        ids_str = ",".join(id_list)
        fetch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=pubmed&id={ids_str}&retmode=xml"
        )
        req_fetch = urllib.request.Request(fetch_url, headers={"User-Agent": "MedVerifyAI/2.0"})
        with urllib.request.urlopen(req_fetch, timeout=15) as resp_xml:
            xml_data = resp_xml.read()

        root = ET.fromstring(xml_data)
        articles = []

        for article_elem in root.findall(".//PubmedArticle"):
            try:
                pmid = article_elem.findtext(".//PMID")
                title = article_elem.findtext(".//ArticleTitle") or ""
                abstract_texts = []
                for abs_text in article_elem.findall(".//AbstractText"):
                    label = abs_text.get("Label")
                    text = abs_text.text or ""
                    if text:
                        if label:
                            abstract_texts.append(f"{label}: {text}")
                        else:
                            abstract_texts.append(text)

                abstract_combined = " ".join(abstract_texts).strip()
                pub_year = 2024
                year_elem = article_elem.find(".//Journal/JournalIssue/PubDate/Year")
                if year_elem is not None and year_elem.text and year_elem.text.isdigit():
                    pub_year = int(year_elem.text)

                if title and len(abstract_combined) > 80:
                    articles.append({
                        "pmid": pmid, "title": title.rstrip("."),
                        "abstract": abstract_combined, "pub_year": pub_year,
                    })
            except Exception:
                continue
        return articles
    except Exception as e:
        print(f"    [!] PubMed error: {e}")
        return []


def main():
    import pandas as pd

    print("=" * 80)
    print("MEDVERIFY AI -- REBUILDING COMPREHENSIVE FAISS KNOWLEDGE BASE (V2)")
    print("=" * 80)

    # Back up old index
    for fname in ["faiss_index.bin", "vector_metadata.json"]:
        src = os.path.join(VECTOR_STORE_DIR, fname)
        dst = os.path.join(BACKUP_DIR, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            import shutil
            shutil.copy2(src, dst)
            print(f"  [Backup] {fname} -> backup_v1/")

    all_chunks = []
    chunk_id = 0
    seen_texts = set()

    # =========================================================================
    # SOURCE 1: PubHealth Explanations as Evidence Passages
    # =========================================================================
    print("\n[SOURCE 1] Ingesting PubHealth fact-check explanations as evidence...")
    df = pd.read_csv(CSV_PATH)
    ph_added = 0

    for _, row in df.iterrows():
        cat = canonicalize_topic(row.get("disease_topic"))
        if cat is None:
            continue

        explanation = str(row.get("explanation", ""))
        main_text = str(row.get("main_text", ""))
        claim_text = str(row.get("claim", ""))
        label = str(row.get("label", ""))
        sources = str(row.get("sources", ""))
        verdict = VERDICT_MAP.get(label, "Neutral")

        # Use explanation as primary evidence (NOT the claim itself)
        if len(explanation) > 80:
            text_hash = hashlib.md5(explanation[:200].encode()).hexdigest()
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                tier = infer_source_tier(explanation, sources)
                all_chunks.append({
                    "vector_id": chunk_id,
                    "chunk_text": explanation[:800],
                    "disease_category": cat,
                    "source_name": "PubHealth_Explanation",
                    "source_tier": tier,
                    "verdict_stance": verdict,
                    "publication_year": 2020,
                    "doc_id": f"ph-exp-{row.get('claim_id', chunk_id)}",
                })
                chunk_id += 1
                ph_added += 1

        # Use main_text excerpt as secondary evidence (first 600 chars)
        if len(main_text) > 200:
            mt_excerpt = main_text[:600]
            text_hash = hashlib.md5(mt_excerpt[:200].encode()).hexdigest()
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                tier = infer_source_tier(main_text, sources)
                all_chunks.append({
                    "vector_id": chunk_id,
                    "chunk_text": mt_excerpt,
                    "disease_category": cat,
                    "source_name": "PubHealth_MainText",
                    "source_tier": tier,
                    "verdict_stance": verdict,
                    "publication_year": 2019,
                    "doc_id": f"ph-mt-{row.get('claim_id', chunk_id)}",
                })
                chunk_id += 1
                ph_added += 1

    print(f"  [OK] Added {ph_added} evidence passages from PubHealth explanations + main_text")

    # =========================================================================
    # SOURCE 2: Real PubMed Abstracts via NCBI E-utilities
    # =========================================================================
    print("\n[SOURCE 2] Fetching real PubMed abstracts across 22 categories...")
    pm_added = 0
    pm_seen_pmids = set()

    for cat_idx, (cat, queries) in enumerate(PUBMED_QUERIES.items(), 1):
        cat_count = 0
        for q_idx, query in enumerate(queries, 1):
            articles = fetch_pubmed_abstracts(query, max_results=8)
            for art in articles:
                if art["pmid"] in pm_seen_pmids:
                    continue
                pm_seen_pmids.add(art["pmid"])

                abstract = art["abstract"]
                tier = infer_source_tier(abstract)

                all_chunks.append({
                    "vector_id": chunk_id,
                    "chunk_text": abstract[:800],
                    "disease_category": cat,
                    "source_name": "PubMed/NCBI",
                    "source_tier": tier,
                    "verdict_stance": "Neutral",
                    "publication_year": art["pub_year"],
                    "doc_id": f"pmid-{art['pmid']}",
                    "title": art["title"],
                })
                chunk_id += 1
                pm_added += 1
                cat_count += 1

            time.sleep(0.35)  # NCBI rate limit

        print(f"  [{cat_idx:2d}/22] {cat:<35s}: +{cat_count} PubMed abstracts")

    print(f"  [OK] Added {pm_added} PubMed abstracts total")

    # =========================================================================
    # RE-INDEX VECTOR IDs
    # =========================================================================
    for i, ch in enumerate(all_chunks):
        ch["vector_id"] = i

    # =========================================================================
    # REPORT CATEGORY DISTRIBUTION
    # =========================================================================
    cat_dist = Counter(ch["disease_category"] for ch in all_chunks)
    source_dist = Counter(ch["source_name"] for ch in all_chunks)
    tier_dist = Counter(ch["source_tier"] for ch in all_chunks)

    print(f"\n{'=' * 80}")
    print(f"KNOWLEDGE BASE V2 SUMMARY: {len(all_chunks)} evidence chunks")
    print(f"{'=' * 80}")
    print(f"\n--- By Disease Category ---")
    for cat in DISEASE_CATEGORIES:
        cnt = cat_dist.get(cat, 0)
        print(f"  {cat:<35s}: {cnt:5d}")
    print(f"\n--- By Source ---")
    for src, cnt in sorted(source_dist.items(), key=lambda x: -x[1]):
        print(f"  {src:<35s}: {cnt:5d}")
    print(f"\n--- By Source Tier ---")
    for tier, cnt in sorted(tier_dist.items(), key=lambda x: -x[1]):
        print(f"  {tier:<40s}: {cnt:5d}")

    # =========================================================================
    # BUILD FAISS INDEX
    # =========================================================================
    print(f"\n[INDEXING] Building FAISS vector index over {len(all_chunks)} chunks...")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_dim = model.get_sentence_embedding_dimension()

    texts = [ch["chunk_text"] for ch in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True, batch_size=64)
    embeddings_np = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings_np)

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"  [OK] FAISS index saved: {index.ntotal} vectors, dim={embedding_dim}")
    print(f"  [OK] Metadata saved: {METADATA_PATH}")

    # Quick test query
    print(f"\n[TEST] Running test similarity search...")
    test_queries = [
        ("Does the MMR vaccine cause autism?", "Vaccination"),
        ("Is metformin effective for type 2 diabetes?", "Diabetes"),
        ("Does smoking cause lung cancer?", "Lung Cancer"),
    ]
    for tq, expected_cat in test_queries:
        qv = model.encode([tq], normalize_embeddings=True).astype("float32")
        scores, indices_result = index.search(qv, 3)
        print(f"\n  Query: '{tq}'")
        for rank, (score, idx) in enumerate(zip(scores[0], indices_result[0])):
            if idx < len(all_chunks):
                ch = all_chunks[idx]
                print(f"    #{rank+1} (sim={score:.3f}) [{ch['disease_category']}] "
                      f"[{ch['source_name']}] {ch['chunk_text'][:80]}...")

    print(f"\n{'=' * 80}")
    print(f"KNOWLEDGE BASE V2 REBUILD COMPLETE")
    print(f"  Total vectors: {index.ntotal}")
    print(f"  Categories covered: {len(cat_dist)}/22")
    print(f"  Min per category: {min(cat_dist.values()) if cat_dist else 0}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
