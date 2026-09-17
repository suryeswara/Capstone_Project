export type DiseaseCategory = 'Cardiovascular Disease' | 'Vaccination' | 'Diabetes' | 'General Medicine'

export type Verdict = 'Supported' | 'Contradicted' | 'Insufficient Evidence' | 'Mixed' | 'TRUE' | 'FALSE' | 'MIXTURE' | 'UNPROVEN'

export type SourceType = 'WHO Guideline' | 'CDC Guideline' | 'Systematic Review' | 'RCT' | 'PubMed Article' | 'Cohort Study' | 'Preprint' | string

export type EvidenceStance = 'supporting' | 'contradicting' | 'neutral'

export interface EvidenceItem {
  id: string
  title: string
  source: SourceType
  authors?: string
  year?: number
  pubYear?: number
  doi?: string
  pmid?: string
  similarity?: number
  reliabilityScore: number // R_i (0-100 or 0-1)
  applicabilityScore?: number // P_i (0-100 or 0-1)
  finalWeight?: number // W_i = R_i * P_i
  pAge?: number
  pSex?: number
  pCondition?: number
  pRegion?: number
  populationMatchType?: 'MATCHED' | 'PARTIAL' | 'MISMATCHED' | 'UNKNOWN'
  stance: EvidenceStance
  abstract?: string
  abstractChunk?: string
  url?: string
}

export type PipelineStageStatus = 'pending' | 'active' | 'complete' | 'flagged'

export interface PipelineStage {
  id: string
  label: string
  description: string
  status: PipelineStageStatus
  detail?: string
  durationMs: number
}

export interface FaithfulnessSentence {
  id: string
  text: string
  status: 'verified' | 'unsupported' | 'contradiction' | 'SUPPORTED' | 'PARTIALLY_SUPPORTED' | 'UNSUPPORTED' | 'CERTAINTY_ESCALATION'
  confidence: number
  evidenceIds: string[]
}

export interface CredibilityBreakdown {
  overall: number
  evidenceConfidence: number
  consensusConfidence: number
  sourceQuality: number
  faithfulnessConfidence: number
}

export interface ConsensusPoint {
  year: number
  supporting: number
  contradicting: number
  neutral: number
}

export interface VerificationRecord {
  id: string
  claim: string
  disease: DiseaseCategory
  submittedAt: string
  verdict: Verdict
  systemConfidence?: number
  credibility: CredibilityBreakdown
  evidence: EvidenceItem[]
  consensusTimeline: ConsensusPoint[]
  explanation: FaithfulnessSentence[]
  modelVersion: string
  bookmarked?: boolean
}

