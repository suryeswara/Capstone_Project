"""
MedVerify AI - Stage 7: Hybrid Retrieval & Evidence Ranking Engine

This module implements:
1. FAISS Vector Similarity Search (Static Knowledge Base)
2. Live PubMed NCBI E-utilities API Retrieval
3. Calibrated Reliability Scoring Function (R_i)
4. Evidence Ranking & Merging Pipeline
"""

import os
import json
import math
import time
import logging
import requests
import numpy as np
import faiss
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RELIABILITY SCORING CONSTANTS (Validated in Stage 0)
# ---------------------------------------------------------------------------

SOURCE_TIERS = {
    "WHO Guideline": 1.00,
    "CDC Guideline": 0.98,
    "WHO/CDC Guideline": 0.99,
    "Systematic Review / Meta-Analysis": 0.95,
    "Randomized Controlled Trial (RCT)": 0.90,
    "Cohort Study / Observational": 0.75,
    "Case Report": 0.55,
    "Preprint (bioRxiv/medRxiv)": 0.30,
    "PubMed Article": 0.80,
    "Unverified Blog / Opinion": 0.05,
}

DISEASE_DECAY_RATES = {
    "Vaccination": 0.10,
    "Cardiovascular Disease": 0.05,
    "Diabetes": 0.03,
}


def calculate_reliability_score(
    source_tier: str,
    pub_year: int,
    disease_category: str = "Diabetes",
    is_peer_reviewed: bool = True,
    current_year: int = 2026,
) -> float:
    """
    Calibrated reliability score R_i in [0.0, 1.0].
    R_i = TierWeight * exp(-lambda * age) * PeerReviewFactor
    """
    tier_weight = SOURCE_TIERS.get(source_tier, 0.50)
    decay_rate = DISEASE_DECAY_RATES.get(disease_category, 0.05)
    age = max(0, current_year - pub_year)
    recency_factor = math.exp(-decay_rate * age)
    peer_review_factor = 1.00 if is_peer_reviewed else 0.70
    raw = tier_weight * recency_factor * peer_review_factor
    return round(min(1.0, max(0.0, raw)), 4)


# ---------------------------------------------------------------------------
# FAISS VECTOR RETRIEVER (Static Knowledge Base - Stage 5 Index)
# ---------------------------------------------------------------------------

class FAISSRetriever:
    """Loads the Stage 5 FAISS index and metadata for similarity search."""

    def __init__(self, vector_store_dir: str):
        index_path = os.path.join(vector_store_dir, "faiss_index.bin")
        meta_path = os.path.join(vector_store_dir, "vector_metadata.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Run Stage 5 first.")

        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info(f"FAISS Retriever loaded: {self.index.ntotal} vectors, dim={self.index.d}")

    def search(self, query: str, top_k: int = 5, disease_filter: Optional[str] = None) -> List[Dict]:
        """Retrieve top-k similar evidence chunks from the static vector store."""
        query_vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_vec, min(top_k * 3, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            # Optional disease category pre-filter
            if disease_filter and meta.get("disease_category") != disease_filter:
                continue
            results.append({
                "source": "FAISS_STATIC",
                "title": meta.get("raw_claim", ""),
                "chunk_text": meta.get("chunk_text", ""),
                "source_tier": meta.get("source_tier", "PubMed Article"),
                "pub_year": meta.get("publication_year", 2024),
                "disease_category": meta.get("disease_category", ""),
                "doc_id": meta.get("doc_id", ""),
                "cosine_similarity": float(score),
                "reliability_score": meta.get("reliability_score", 0.50),
                "stance": meta.get("verdict_stance", "supporting"),
            })
            if len(results) >= top_k:
                break
        return results


# ---------------------------------------------------------------------------
# PUBMED LIVE RETRIEVER (NCBI E-utilities API)
# ---------------------------------------------------------------------------

class PubMedRetriever:
    """Fetches live research abstracts from NCBI PubMed E-utilities API."""

    BASE_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    BASE_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    BASE_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    TIMEOUT = 8  # seconds

    def _fetch_abstracts(self, id_list: List[str]) -> Dict[str, str]:
        """
        Batch fetch study abstract text via NCBI efetch XML API.
        Returns a mapping of pmid -> abstract_text.
        """
        abstract_map = {}
        if not id_list:
            return abstract_map

        try:
            params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml",
            }
            resp = requests.get(self.BASE_EFETCH, params=params, timeout=self.TIMEOUT)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for article in root.findall(".//PubmedArticle"):
                    pmid_elem = article.find(".//MedlineCitation/PMID")
                    if pmid_elem is not None and pmid_elem.text:
                        pmid = pmid_elem.text.strip()
                        abstract_parts = [
                            elem.text.strip()
                            for elem in article.findall(".//Abstract/AbstractText")
                            if elem.text and elem.text.strip()
                        ]
                        if abstract_parts:
                            full_abstract = " ".join(abstract_parts)
                            # Keep first 600 characters for concise NLI premise
                            abstract_map[pmid] = full_abstract[:600]
        except Exception as e:
            logger.warning(f"Could not fetch full abstracts via efetch: {e}. Falling back to titles.")

        return abstract_map

    def search(self, query: str, disease_category: str = "", max_results: int = 5) -> List[Dict]:
        """Search PubMed and retrieve article summaries and abstracts."""
        results = []
        try:
            # Step 1: Search for PMIDs
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": max_results,
                "sort": "relevance",
            }
            resp = requests.get(self.BASE_SEARCH, params=search_params, timeout=self.TIMEOUT)
            resp.raise_for_status()
            id_list = resp.json().get("esearchresult", {}).get("idlist", [])

            if not id_list:
                logger.warning(f"PubMed returned 0 results for query: {query[:50]}")
                return results

            # Step 2: Fetch article summaries (metadata, titles, dates)
            ids_str = ",".join(id_list)
            fetch_params = {"db": "pubmed", "id": ids_str, "retmode": "json"}
            fetch_resp = requests.get(self.BASE_FETCH, params=fetch_params, timeout=self.TIMEOUT)
            fetch_resp.raise_for_status()
            result_dict = fetch_resp.json().get("result", {})

            # Step 2.5: Batch fetch genuine abstract paragraphs via efetch
            abstract_map = self._fetch_abstracts(id_list)

            for pmid in id_list:
                article = result_dict.get(str(pmid), {})
                title = article.get("title", "").strip()
                if not title:
                    continue

                # Extract publication year
                pubdate = article.get("pubdate", "2024")
                pub_year = int(pubdate.split()[0]) if pubdate.split() and pubdate.split()[0].isdigit() else 2024

                # Determine source tier from publication type
                pub_types = [pt.lower() for pt in article.get("pubtype", [])]
                if "meta-analysis" in pub_types or "systematic review" in pub_types:
                    source_tier = "Systematic Review / Meta-Analysis"
                elif "randomized controlled trial" in pub_types:
                    source_tier = "Randomized Controlled Trial (RCT)"
                elif "clinical trial" in pub_types:
                    source_tier = "Randomized Controlled Trial (RCT)"
                elif "review" in pub_types:
                    source_tier = "Cohort Study / Observational"
                else:
                    source_tier = "PubMed Article"

                doi = ""
                article_ids = article.get("articleids", [])
                for aid in article_ids:
                    if aid.get("idtype") == "doi":
                        doi = aid.get("value", "")
                        break

                reliability = calculate_reliability_score(
                    source_tier=source_tier,
                    pub_year=pub_year,
                    disease_category=disease_category,
                    is_peer_reviewed=True,
                )

                # Include abstract if available for NLI stance detection
                abstract_text = abstract_map.get(str(pmid), "")
                if abstract_text:
                    chunk_text = f"PubMed Article (PMID:{pmid}): {title}. Abstract: {abstract_text}"
                else:
                    chunk_text = f"PubMed Article (PMID:{pmid}): {title}"

                results.append({
                    "source": "PUBMED_LIVE",
                    "title": title.rstrip("."),
                    "chunk_text": chunk_text,
                    "abstract": abstract_text,
                    "source_tier": source_tier,
                    "pub_year": pub_year,
                    "disease_category": disease_category,
                    "doc_id": f"pmid-{pmid}",
                    "doi": doi,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "cosine_similarity": 0.0,  # Not vector-matched
                    "reliability_score": reliability,
                    "stance": "supporting",  # Default; refined by Consensus Engine (Stage 8)
                })

        except requests.exceptions.Timeout:
            logger.warning("PubMed API request timed out. Falling back to static FAISS only.")
        except requests.exceptions.RequestException as e:
            logger.warning(f"PubMed API error: {e}. Falling back to static FAISS only.")

        return results


# ---------------------------------------------------------------------------
# HYBRID RETRIEVAL & EVIDENCE RANKING ENGINE
# ---------------------------------------------------------------------------

class HybridRetrievalEngine:
    """
    Combines FAISS static retrieval + PubMed live retrieval,
    then ranks all evidence by calibrated reliability score R_i.
    """

    def __init__(self, vector_store_dir: str):
        self.faiss_retriever = FAISSRetriever(vector_store_dir)
        self.pubmed_retriever = PubMedRetriever()

    def retrieve_and_rank(
        self,
        claim_text: str,
        disease_category: str,
        faiss_top_k: int = 5,
        pubmed_max: int = 3,
    ) -> List[Dict]:
        """
        Execute hybrid retrieval and return evidence ranked by reliability score.
        """
        # 1. FAISS Static Retrieval
        faiss_results = self.faiss_retriever.search(
            query=claim_text, top_k=faiss_top_k, disease_filter=disease_category
        )
        logger.info(f"FAISS returned {len(faiss_results)} results for '{claim_text[:40]}...'")

        # 2. PubMed Live Retrieval
        pubmed_results = self.pubmed_retriever.search(
            query=claim_text, disease_category=disease_category, max_results=pubmed_max
        )
        logger.info(f"PubMed returned {len(pubmed_results)} results for '{claim_text[:40]}...'")

        # 3. Merge & Deduplicate (by doc_id)
        seen_ids = set()
        merged = []
        for item in faiss_results + pubmed_results:
            doc_id = item.get("doc_id", "")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                merged.append(item)

        # 4. Recalculate reliability scores for all merged evidence
        for item in merged:
            item["reliability_score"] = calculate_reliability_score(
                source_tier=item.get("source_tier", "PubMed Article"),
                pub_year=item.get("pub_year", 2024),
                disease_category=disease_category,
                is_peer_reviewed=True,
            )

        # 5. Rank by reliability score (descending)
        ranked = sorted(merged, key=lambda x: x["reliability_score"], reverse=True)

        return ranked
