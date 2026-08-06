export type DiseaseCategory = 'Cardiovascular Disease' | 'Vaccination' | 'Diabetes'

export type Verdict = 'Supported' | 'Contradicted' | 'Insufficient Evidence' | 'Mixed'

export type SourceType = 'WHO Guideline' | 'CDC Guideline' | 'Systematic Review' | 'RCT' | 'PubMed Article' | 'Cohort Study' | 'Preprint'

export type EvidenceStance = 'supporting' | 'contradicting' | 'neutral'

export interface EvidenceItem {
  id: string
  title: string
  source: SourceType
  authors: string
  year: number
  doi: string
  reliabilityScore: number // 0-100, source-quality weighting
  stance: EvidenceStance
  abstract: string
  url: string
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
  status: 'verified' | 'unsupported' | 'contradiction'
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
  credibility: CredibilityBreakdown
  evidence: EvidenceItem[]
  consensusTimeline: ConsensusPoint[]
  explanation: FaithfulnessSentence[]
  modelVersion: string
  bookmarked?: boolean
}
