import type { PipelineStage } from '@/types'

export const PIPELINE_TEMPLATE: Omit<PipelineStage, 'status'>[] = [
  { id: 'understand', label: 'Claim Understanding', description: 'Parsing claim structure and extracting the core medical assertion.', durationMs: 900 },
  { id: 'classify', label: 'Disease Classification', description: 'Routing the claim to the relevant clinical domain via BioBERT.', durationMs: 700 },
  { id: 'retrieve', label: 'Evidence Retrieval', description: 'Hybrid search across the static WHO/CDC corpus and live PubMed.', durationMs: 1400 },
  { id: 'rank', label: 'Evidence Quality Ranking', description: 'Scoring each source by study design and reliability.', durationMs: 1000 },
  { id: 'consensus', label: 'Scientific Consensus Analysis', description: 'Tallying supporting, contradicting, and neutral findings.', durationMs: 900 },
  { id: 'explain', label: 'AI Explanation Generation', description: 'Drafting a plain-language explanation grounded in the retrieved evidence.', durationMs: 1300 },
  { id: 'faithfulness', label: 'Faithfulness Verification', description: 'Checking every explanation sentence against evidence with an NLI entailment model.', durationMs: 1100 },
  { id: 'score', label: 'Credibility Score', description: 'Combining evidence, consensus, source quality, and faithfulness into one score.', durationMs: 600 },
  { id: 'verdict', label: 'Evidence-Backed Final Verdict', description: 'Presenting the verdict with full traceability to its evidence.', durationMs: 500 },
]

export function freshStages(): PipelineStage[] {
  return PIPELINE_TEMPLATE.map((s) => ({ ...s, status: 'pending' as const }))
}
