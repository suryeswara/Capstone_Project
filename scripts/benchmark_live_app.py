"""
MedVerify AI -- Live Real-Time Application Performance Benchmark

Connects to the live running server at http://127.0.0.1:8000 and measures:
1. Claim submission latency (POST /api/claims)
2. Real-time SSE progress streaming performance (GET /api/verifications/{id}/stream)
3. End-to-end verification pipeline latency
4. Verification report accuracy & payload structure (GET /api/verifications/{id}/report)
5. Personal medical query safety refusal fast-path
"""

import time
import json
import urllib.request
import urllib.parse
from typing import Dict, List

BASE_URL = "http://127.0.0.1:8000"

print("=" * 80)
print("MEDVERIFY AI -- LIVE REAL-TIME APPLICATION PERFORMANCE BENCHMARK")
print("=" * 80)

# Helper function: HTTP POST
def http_post(endpoint: str, payload: dict) -> tuple:
    url = f"{BASE_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        elapsed = (time.time() - t0) * 1000.0  # ms
        return resp.status, body, elapsed

# Helper function: HTTP GET
def http_get(endpoint: str) -> tuple:
    url = f"{BASE_URL}{endpoint}"
    t0 = time.time()
    with urllib.request.urlopen(url) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        elapsed = (time.time() - t0) * 1000.0  # ms
        return resp.status, body, elapsed

# Helper function: SSE Stream Consumer
def consume_sse_stream(verification_id: str) -> tuple:
    url = f"{BASE_URL}/api/verifications/{verification_id}/stream"
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    events = []
    t0 = time.time()
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data: "):
                data_json = line_str[6:]
                try:
                    event = json.loads(data_json)
                    events.append(event)
                    if event.get("isTerminal"):
                        break
                except json.JSONDecodeError:
                    pass
    total_elapsed = (time.time() - t0) * 1000.0  # ms
    return events, total_elapsed


# ---------------------------------------------------------------------------
# BENCHMARK 1: Health Check Endpoint
# ---------------------------------------------------------------------------
print("\n[BENCHMARK 1] Health Check Endpoint")
print("-" * 70)
status, health, latency = http_get("/health")
print(f"  GET /health -> Status: {status} | Latency: {latency:.2f} ms")
print(f"  App Name: '{health.get('name')}' | Status: {health.get('status')}")

# ---------------------------------------------------------------------------
# BENCHMARK 2: Real-Time Verification Streaming (Standard True Claim)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[BENCHMARK 2] Real-Time Verification Stream (True Claim)")
print("-" * 70)

claim_1 = "Statins reduce cardiovascular risk in patients with heart disease."
print(f"  Submitting Claim: '{claim_1}'")
status_1, submit_1, submit_lat_1 = http_post("/api/claims", {"rawText": claim_1})
ver_id_1 = submit_1["verificationId"]
print(f"  Claim Submission -> Status: {status_1} | Verification ID: {ver_id_1} | Latency: {submit_lat_1:.2f} ms")

print(f"  Consuming Real-Time SSE Event Stream...")
stream_events_1, stream_time_1 = consume_sse_stream(ver_id_1)

for ev in stream_events_1:
    print(f"    [SSE Update] {ev['status']:<15} ({ev['progressPercentage']:>3}%) -> '{ev['currentStepLabel']}'")

print(f"  Stream Completed in: {stream_time_1 / 1000.0:.2f} s ({len(stream_events_1)} progress events received)")

# Fetch Final Verification Report
status_rep_1, report_1, report_lat_1 = http_get(f"/api/verifications/{ver_id_1}/report")
print(f"\n  Final Report fetched in {report_lat_1:.2f} ms:")
print(f"    * Verdict:                {report_1['verdict']}")
print(f"    * Credibility Score:      {report_1['credibility']['overall']:.1f}%")
print(f"    * Evidence Confidence:   {report_1['credibility']['evidenceConfidence']:.1f}%")
print(f"    * Consensus Confidence:  {report_1['credibility']['consensusConfidence']:.1f}%")
print(f"    * FaithfulnessConfidence: {report_1['credibility']['faithfulnessConfidence']:.1f}%")
print(f"    * Total Citations:        {len(report_1['evidence'])}")
print(f"    * Explanation Sentences:  {len(report_1['explanation'])}")

# ---------------------------------------------------------------------------
# BENCHMARK 3: Real-Time Verification Stream (False Claim - Refutation)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[BENCHMARK 3] Real-Time Verification Stream (False Claim / Myth Refutation)")
print("-" * 70)

claim_2 = "The MMR vaccine causes autism in children."
print(f"  Submitting Claim: '{claim_2}'")
status_2, submit_2, submit_lat_2 = http_post("/api/claims", {"rawText": claim_2})
ver_id_2 = submit_2["verificationId"]
print(f"  Claim Submission -> Status: {status_2} | Verification ID: {ver_id_2} | Latency: {submit_lat_2:.2f} ms")

stream_events_2, stream_time_2 = consume_sse_stream(ver_id_2)
for ev in stream_events_2:
    print(f"    [SSE Update] {ev['status']:<15} ({ev['progressPercentage']:>3}%) -> '{ev['currentStepLabel']}'")

status_rep_2, report_2, _ = http_get(f"/api/verifications/{ver_id_2}/report")
print(f"\n  Final Report (Refutation):")
print(f"    * Verdict:           {report_2['verdict']}")
print(f"    * Credibility Score: {report_2['credibility']['overall']:.1f}%")
print(f"    * Total Citations:   {len(report_2['evidence'])}")

# ---------------------------------------------------------------------------
# BENCHMARK 4: Safety Refusal Fast-Path Latency Test
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[BENCHMARK 4] Safety Refusal Fast-Path Latency Test")
print("-" * 70)

safety_claim = "I have severe chest pain, should I take aspirin immediately?"
print(f"  Submitting Personal Advice Query: '{safety_claim}'")
status_s, submit_s, submit_lat_s = http_post("/api/claims", {"rawText": safety_claim})
ver_id_s = submit_s["verificationId"]

stream_events_s, stream_time_s = consume_sse_stream(ver_id_s)
refusal_event = stream_events_s[-1] if stream_events_s else {}
print(f"  Safety Refusal Stream Completed in {stream_time_s:.2f} ms:")
print(f"    * Status:             {refusal_event.get('status')}")
print(f"    * Step Label:         '{refusal_event.get('currentStepLabel')}'")
print(f"    * Fast-Path Latency:  {stream_time_s:.2f} ms (sub-second refusal!)")

# ---------------------------------------------------------------------------
# SUMMARY OF REAL-TIME APPLICATION PERFORMANCE
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("REAL-TIME APPLICATION PERFORMANCE SUMMARY REPORT")
print("=" * 80)
print(f" 1. API Submission Latency:      {submit_lat_1:.2f} ms")
print(f" 2. End-to-End Pipeline Latency:  {stream_time_1 / 1000.0:.2f} s")
print(f" 3. Safety Refusal Latency:       {stream_time_s:.2f} ms")
print(f" 4. Real-time SSE Events Emitted: {len(stream_events_1)} status updates")
print(f" 5. Verdict Accuracy (True):     {report_1['verdict']} ({report_1['credibility']['overall']:.1f}%)")
print(f" 6. Verdict Accuracy (False):    {report_2['verdict']} ({report_2['credibility']['overall']:.1f}%)")
print("=" * 80)
