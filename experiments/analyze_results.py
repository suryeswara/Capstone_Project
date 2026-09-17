import json

r = json.load(open('Reports/full_baseline_comparison.json'))
print('=== DATASET ===')
print('Total items:', r['dataset']['total_items'])
print('Label distribution:', r['dataset']['label_distribution'])
print()

for bname in ['B_minus_1_Keyword', 'B0_LLM_Only', 'B1_Conventional_RAG', 'B2_Reliability_RAG', 'B3_Full_MedVerify']:
    b = r['baselines'].get(bname, {})
    print(f'=== {bname} ===')
    print(f"  Accuracy: {b.get('accuracy',0)*100:.2f}%  Macro-F1: {b.get('macro_f1',0)*100:.2f}%")
    if 'per_class_f1' in b:
        for k, v in b['per_class_f1'].items():
            print(f'    {k}: {v:.4f}')
    if 'confusion_matrix' in b:
        print(f'  Confusion matrix: {b["confusion_matrix"]}')
    print()

print('=== Verdict Change Analysis ===')
print(json.dumps(r['verdict_change_analysis'], indent=2))
print()

# Check predictions
print('=== Sample Predictions (first 15) ===')
for d in r['detailed_results'][:15]:
    print(f"  gold={d['gold']:10s} bm1={d.get('b_minus_1','?'):10s} b0={d['b0']:10s} b1={d['b1']:10s} b2={d['b2']:10s} b3={d['b3']:10s}")

# Count prediction distributions
from collections import Counter
preds_b0 = Counter(d['b0'] for d in r['detailed_results'])
preds_b3 = Counter(d['b3'] for d in r['detailed_results'])
print()
print('=== B0 Prediction Distribution ===')
for k, v in sorted(preds_b0.items()):
    print(f'  {k}: {v}')
print('=== B3 Prediction Distribution ===')
for k, v in sorted(preds_b3.items()):
    print(f'  {k}: {v}')
