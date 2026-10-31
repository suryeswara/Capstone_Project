import type { PipelineStage } from '@/types'

export const PIPELINE_TEMPLATE: Omit<PipelineStage, 'status'>[] = [
  { id: 'validate', label: 'Claim validation', description: 'Length and safety screening of the submitted text.', durationMs: 0 },
  { id: 'gate', label: 'Medical-claim gate', description: 'NLI + keyword + MedCPT-probe stacker decides whether this is a medical claim.', durationMs: 0 },
  { id: 'retrieve', label: 'Evidence retrieval', description: 'Local FAISS fact-checks and abstracts (static), live PubMed, or both (hybrid).', durationMs: 0 },
  { id: 'rerank', label: 'Evidence reranking', description: 'MonoT5 scores every claim\u2013document pair and keeps the best.', durationMs: 0 },
  { id: 'classify', label: 'Evidence classification', description: 'DeBERTa + BiomedBERT ensemble: SUPPORT / CONTRADICT / NEI for each pair.', durationMs: 0 },
  { id: 'meta', label: 'Meta-model verdict', description: 'Static, live or hybrid meta-model combines the evidence into one verdict.', durationMs: 0 },
  { id: 'explain', label: 'Explanation & audit', description: 'Grounded explanation, better-claim suggestions and faithfulness check.', durationMs: 0 },
  { id: 'verdict', label: 'Final report', description: 'Verdict, probabilities, review flag, evidence cards and sources.', durationMs: 0 },
]

export function freshStages(): PipelineStage[] {
  return PIPELINE_TEMPLATE.map((s) => ({ ...s, status: 'pending' as const }))
}

/** Which stage is active for a backend status (see STAGE_STATUS in verifications.py). */
export function stageIndexFor(status: string, progress: number): number {
  switch (status) {
    case 'CREATED': return 0
    case 'EXTRACTING': return progress < 10 ? 0 : 1
    case 'RETRIEVING': return 2
    case 'RANKING': return 3
    case 'CLASSIFYING': return 4
    case 'CONSENSUS': return 5
    case 'GENERATING':
    case 'VERIFYING': return 6
    default: return PIPELINE_TEMPLATE.length - 1
  }
}
