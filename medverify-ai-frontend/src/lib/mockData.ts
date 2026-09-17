import type { VerificationRecord, EvidenceItem, FaithfulnessSentence } from '@/types'

const mkEvidence = (items: Omit<EvidenceItem, 'id'>[]): EvidenceItem[] =>
  items.map((it, i) => ({ ...it, id: `ev-${i}-${(it.doi ?? '').replace(/[^a-z0-9]/gi, '')}` }))


export const EXAMPLE_CLAIMS: { text: string; disease: string }[] = [
  { text: 'The seasonal flu vaccine can give you the flu.', disease: 'Vaccination' },
  { text: 'Statins reduce the risk of recurrent heart attacks in people with existing heart disease.', disease: 'Cardiovascular Disease' },
  { text: 'Type 2 diabetes can be fully reversed by intermittent fasting alone.', disease: 'Diabetes' },
  { text: 'The MMR vaccine causes autism.', disease: 'Vaccination' },
  { text: 'Daily low-dose aspirin should be taken by all adults over 50 to prevent heart attacks.', disease: 'Cardiovascular Disease' },
]

const vaccineEvidence = mkEvidence([
  {
    title: 'Inactivated influenza vaccines and adverse event surveillance, a decade of data',
    source: 'Systematic Review',
    authors: 'Chen K. et al.',
    year: 2022,
    doi: '10.1001/jama.2022.10234',
    reliabilityScore: 95,
    stance: 'contradicting',
    abstract:
      'A pooled analysis across 41 surveillance studies found no mechanism by which inactivated influenza vaccine, which contains no live virus, can cause influenza infection. Reported post-vaccination illness is attributable to concurrent circulating respiratory viruses or the immune response itself, not the vaccine antigen.',
    url: 'https://pubmed.ncbi.nlm.nih.gov/example1',
  },
  {
    title: 'WHO position paper: influenza vaccines',
    source: 'WHO Guideline',
    authors: 'World Health Organization',
    year: 2023,
    doi: '10.who/ip.2023.influenza',
    reliabilityScore: 98,
    stance: 'contradicting',
    abstract:
      'Inactivated seasonal influenza vaccines cannot cause influenza disease because they do not contain live virus. Mild, self-limited reactions such as soreness or low-grade fever reflect a normal immune response, not infection.',
    url: 'https://who.int/example-flu',
  },
  {
    title: 'CDC guidance: seasonal flu vaccine safety and side effects',
    source: 'CDC Guideline',
    authors: 'Centers for Disease Control and Prevention',
    year: 2024,
    doi: '10.cdc/flu.2024.safety',
    reliabilityScore: 96,
    stance: 'contradicting',
    abstract:
      'Common side effects of flu vaccination are mild and temporary: soreness, low fever, and muscle aches. These reflect immune activation, not infection with influenza virus, and typically resolve within one to two days.',
    url: 'https://cdc.gov/example-flu',
  },
  {
    title: 'Timing of respiratory illness onset relative to influenza vaccination in a cohort of 62,000 adults',
    source: 'Cohort Study',
    authors: 'Alvarez R., Kim S.',
    year: 2021,
    doi: '10.1016/lancet.2021.44821',
    reliabilityScore: 82,
    stance: 'neutral',
    abstract:
      'Illness onset within two weeks of vaccination was tracked across a large adult cohort. Most reported illness episodes were confirmed via PCR to be unrelated circulating respiratory viruses rather than influenza, though the study notes surveillance gaps in self-reported symptom timing.',
    url: 'https://pubmed.ncbi.nlm.nih.gov/example2',
  },
])

const statinEvidence = mkEvidence([
  {
    title: 'Efficacy of statin therapy for secondary prevention of cardiovascular events: a meta-analysis of 26 trials',
    source: 'Systematic Review',
    authors: 'Baigent C. et al.',
    year: 2021,
    doi: '10.1016/S0140-6736(21)00234-1',
    reliabilityScore: 95,
    stance: 'supporting',
    abstract:
      'Across 26 randomized trials totaling over 170,000 participants, statin therapy reduced major coronary events by approximately 25% per 1 mmol/L reduction in LDL cholesterol among patients with established cardiovascular disease.',
    url: 'https://pubmed.ncbi.nlm.nih.gov/example3',
  },
  {
    title: 'Randomized controlled trial of high-intensity statin therapy post-myocardial infarction',
    source: 'RCT',
    authors: 'Nissen S. et al.',
    year: 2019,
    doi: '10.1056/NEJMoa1900001',
    reliabilityScore: 90,
    stance: 'supporting',
    abstract:
      'Patients randomized to high-intensity atorvastatin after myocardial infarction showed a significant reduction in recurrent cardiovascular events compared with moderate-intensity therapy over a 30-month follow-up period.',
    url: 'https://pubmed.ncbi.nlm.nih.gov/example4',
  },
  {
    title: 'AHA/ACC guideline on the management of blood cholesterol',
    source: 'WHO Guideline',
    authors: 'American Heart Association',
    year: 2023,
    doi: '10.aha/guideline.2023.cholesterol',
    reliabilityScore: 97,
    stance: 'supporting',
    abstract:
      'High-intensity statin therapy is recommended for secondary prevention in all patients with clinical atherosclerotic cardiovascular disease, regardless of baseline LDL cholesterol level, given consistent evidence of event reduction.',
    url: 'https://professional.heart.org/example',
  },
  {
    title: 'Muscle-related adverse events in statin users: a real-world cohort',
    source: 'Cohort Study',
    authors: 'Feldman R., Okafor T.',
    year: 2020,
    doi: '10.1136/bmj.2020.55210',
    reliabilityScore: 78,
    stance: 'neutral',
    abstract:
      'Roughly 5-10% of statin users reported muscle-related symptoms in real-world use, higher than rates seen in blinded trials, suggesting a nocebo component; the study did not evaluate cardiovascular event reduction directly.',
    url: 'https://pubmed.ncbi.nlm.nih.gov/example5',
  },
])

const diabetesEvidence = mkEvidence([
  {
    title: 'Remission of type 2 diabetes after intensive lifestyle intervention: the DiRECT trial at 2 years',
    source: 'RCT',
    authors: 'Lean M. et al.',
    year: 2019,
    doi: '10.1016/S2213-8587(19)30068-3',
    reliabilityScore: 88,
    stance: 'neutral',
    abstract:
      'Sustained weight loss through a structured low-calorie diet produced diabetes remission in 36% of participants at two years, but remission required substantial weight loss maintenance, and intermittent fasting specifically was not the tested intervention.',
    url: 'https://pubmed.ncbi.nlm.nih.gov/example6',
  },
  {
    title: 'Intermittent fasting versus continuous calorie restriction for glycemic control: a systematic review',
    source: 'Systematic Review',
    authors: 'Patel N. et al.',
    year: 2022,
    doi: '10.2337/dc22-0456',
    reliabilityScore: 91,
    stance: 'contradicting',
    abstract:
      'Across 11 randomized trials, intermittent fasting produced glycemic improvements similar to continuous calorie restriction but full, durable remission without medication was rare and dependent on individual weight loss magnitude, contradicting claims of fasting as a standalone cure.',
    url: 'https://pubmed.ncbi.nlm.nih.gov/example7',
  },
  {
    title: 'ADA Standards of Care: diabetes remission and lifestyle therapy',
    source: 'CDC Guideline',
    authors: 'American Diabetes Association',
    year: 2024,
    doi: '10.ada/standards.2024.remission',
    reliabilityScore: 96,
    stance: 'contradicting',
    abstract:
      'Remission is possible for some patients through significant, sustained weight loss but is not guaranteed by any single dietary pattern, including intermittent fasting, and requires ongoing monitoring since relapse is common.',
    url: 'https://diabetes.org/example',
  },
  {
    title: 'Case series: intermittent fasting self-reported outcomes in an online cohort',
    source: 'Preprint',
    authors: 'Nguyen H.',
    year: 2023,
    doi: '10.1101/2023.06.02.preprint',
    reliabilityScore: 34,
    stance: 'supporting',
    abstract:
      'Self-reported outcomes from an unverified online cohort claimed high rates of diabetes reversal using time-restricted eating, though the study lacks clinical confirmation, control groups, and standardized diagnostic criteria.',
    url: 'https://medrxiv.org/example-preprint',
  },
])

function faithfulnessFor(sentences: Omit<FaithfulnessSentence, 'id'>[]): FaithfulnessSentence[] {
  return sentences.map((s, i) => ({ ...s, id: `fs-${i}` }))
}

export const VERIFICATIONS: VerificationRecord[] = [
  {
    id: 'ver-001',
    claim: EXAMPLE_CLAIMS[0].text,
    disease: 'Vaccination',
    submittedAt: '2026-07-28T09:14:00Z',
    verdict: 'Contradicted',
    credibility: { overall: 93, evidenceConfidence: 95, consensusConfidence: 90, sourceQuality: 96, faithfulnessConfidence: 92 },
    evidence: vaccineEvidence,
    consensusTimeline: [
      { year: 2021, supporting: 0, contradicting: 3, neutral: 2 },
      { year: 2022, supporting: 0, contradicting: 5, neutral: 3 },
      { year: 2023, supporting: 0, contradicting: 7, neutral: 2 },
      { year: 2024, supporting: 0, contradicting: 9, neutral: 1 },
    ],
    explanation: faithfulnessFor([
      { text: 'The seasonal inactivated influenza vaccine contains no live virus, so it cannot cause an influenza infection.', status: 'verified', confidence: 97, evidenceIds: [vaccineEvidence[0].id, vaccineEvidence[1].id] },
      { text: 'Mild reactions such as soreness or a low-grade fever reflect the immune system responding to the vaccine, not a viral infection.', status: 'verified', confidence: 95, evidenceIds: [vaccineEvidence[1].id, vaccineEvidence[2].id] },
      { text: 'Illness shortly after vaccination is typically an unrelated, already-circulating respiratory virus.', status: 'verified', confidence: 88, evidenceIds: [vaccineEvidence[3].id] },
      { text: 'Some individuals report feeling noticeably worse than a typical vaccine reaction for up to a week.', status: 'unsupported', confidence: 41, evidenceIds: [] },
    ]),
    modelVersion: 'qwen3-8b-instruct · faithfulness-nli-v1.2',
  },
  {
    id: 'ver-002',
    claim: EXAMPLE_CLAIMS[1].text,
    disease: 'Cardiovascular Disease',
    submittedAt: '2026-07-29T14:02:00Z',
    verdict: 'Supported',
    credibility: { overall: 96, evidenceConfidence: 97, consensusConfidence: 95, sourceQuality: 97, faithfulnessConfidence: 96 },
    evidence: statinEvidence,
    consensusTimeline: [
      { year: 2019, supporting: 4, contradicting: 0, neutral: 1 },
      { year: 2021, supporting: 7, contradicting: 0, neutral: 2 },
      { year: 2023, supporting: 10, contradicting: 0, neutral: 2 },
      { year: 2024, supporting: 12, contradicting: 0, neutral: 3 },
    ],
    explanation: faithfulnessFor([
      { text: 'Statin therapy reduces major coronary events by roughly a quarter per unit reduction in LDL cholesterol in patients with existing cardiovascular disease.', status: 'verified', confidence: 96, evidenceIds: [statinEvidence[0].id] },
      { text: 'Randomized trials directly comparing statin intensities after a heart attack confirm fewer recurrent events with higher-intensity therapy.', status: 'verified', confidence: 94, evidenceIds: [statinEvidence[1].id] },
      { text: 'Major cardiology guidelines recommend high-intensity statins for secondary prevention regardless of baseline cholesterol level.', status: 'verified', confidence: 97, evidenceIds: [statinEvidence[2].id] },
      { text: 'Muscle-related side effects are rare and occur in under 1% of patients.', status: 'contradiction', confidence: 62, evidenceIds: [statinEvidence[3].id] },
    ]),
    modelVersion: 'qwen3-8b-instruct · faithfulness-nli-v1.2',
  },
  {
    id: 'ver-003',
    claim: EXAMPLE_CLAIMS[2].text,
    disease: 'Diabetes',
    submittedAt: '2026-07-30T11:41:00Z',
    verdict: 'Mixed',
    credibility: { overall: 71, evidenceConfidence: 74, consensusConfidence: 66, sourceQuality: 80, faithfulnessConfidence: 72 },
    evidence: diabetesEvidence,
    consensusTimeline: [
      { year: 2019, supporting: 1, contradicting: 1, neutral: 2 },
      { year: 2021, supporting: 2, contradicting: 2, neutral: 2 },
      { year: 2022, supporting: 2, contradicting: 4, neutral: 1 },
      { year: 2023, supporting: 3, contradicting: 5, neutral: 1 },
    ],
    explanation: faithfulnessFor([
      { text: 'Sustained, significant weight loss can produce diabetes remission in a meaningful share of patients.', status: 'verified', confidence: 89, evidenceIds: [diabetesEvidence[0].id] },
      { text: 'Intermittent fasting produces glycemic improvements comparable to standard calorie restriction, not a guaranteed cure.', status: 'verified', confidence: 85, evidenceIds: [diabetesEvidence[1].id] },
      { text: 'Fasting alone reliably reverses type 2 diabetes for most people without any other lifestyle change.', status: 'contradiction', confidence: 78, evidenceIds: [diabetesEvidence[2].id] },
      { text: 'Remission achieved this way tends to be permanent and does not require ongoing monitoring.', status: 'unsupported', confidence: 35, evidenceIds: [] },
    ]),
    modelVersion: 'qwen3-8b-instruct · faithfulness-nli-v1.2',
  },
]

export const DISEASE_DISTRIBUTION = [
  { name: 'Vaccination', value: 41, color: '#1B9CB5' },
  { name: 'Cardiovascular', value: 33, color: '#2F6FED' },
  { name: 'Diabetes', value: 26, color: '#1E9E6B' },
]

export const VERIFICATION_TREND = [
  { week: 'Wk 1', verifications: 62, avgCredibility: 81 },
  { week: 'Wk 2', verifications: 74, avgCredibility: 83 },
  { week: 'Wk 3', verifications: 69, avgCredibility: 79 },
  { week: 'Wk 4', verifications: 91, avgCredibility: 85 },
  { week: 'Wk 5', verifications: 103, avgCredibility: 87 },
  { week: 'Wk 6', verifications: 96, avgCredibility: 86 },
  { week: 'Wk 7', verifications: 118, avgCredibility: 89 },
  { week: 'Wk 8', verifications: 134, avgCredibility: 90 },
]

export const SOURCE_USAGE = [
  { source: 'PubMed', count: 412 },
  { source: 'PubMed Central', count: 268 },
  { source: 'WHO', count: 121 },
  { source: 'CDC', count: 98 },
  { source: 'Preprints', count: 19 },
]
