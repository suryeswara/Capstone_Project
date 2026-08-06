import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  ArrowRight,
  ShieldCheck,
  FlaskConical,
  ScaleIcon,
  Sparkles,
  Gauge,
  BookOpenCheck,
  Database,
  GitBranch,
  Check,
  Minus,
} from 'lucide-react'
import { PublicNav } from '@/components/layout/PublicNav'
import { Footer } from '@/components/layout/Footer'
import { Button, Badge, SectionHeading, Card, Eyebrow } from '@/components/ui/Primitives'
import { PipelineVisualizer } from '@/components/pipeline/PipelineVisualizer'
import { CredibilityGauge } from '@/components/credibility/CredibilityGauge'
import { PIPELINE_TEMPLATE } from '@/lib/pipeline'
import type { PipelineStage } from '@/types'

const demoStages: PipelineStage[] = PIPELINE_TEMPLATE.map((s, i) => ({
  ...s,
  status: i < 6 ? 'complete' : i === 6 ? 'active' : 'pending',
}))

const FAQ = [
  {
    q: 'How is this different from a fact-checking chatbot?',
    a: 'A chatbot answers from memory and can sound confident while being wrong. MedVerify AI retrieves evidence first, ranks it by source quality, and — critically — runs a separate faithfulness check confirming every explanation sentence is actually entailed by that evidence before it is shown to you.',
  },
  {
    q: 'What happens when the AI explanation isn\u2019t fully supported?',
    a: 'The sentence is flagged as unsupported or contradictory rather than hidden or silently corrected. You always see which parts of the explanation passed verification and which did not.',
  },
  {
    q: 'Which sources does MedVerify AI use?',
    a: 'A static, curated corpus of WHO and CDC guidelines for the covered disease categories, combined with live retrieval from PubMed and PubMed Central. Every source is scored for reliability based on its study design.',
  },
  {
    q: 'What is currently in scope?',
    a: 'Phase 1 covers three disease categories — cardiovascular disease, vaccination, and diabetes — for English-language text claims. Multimodal input and expanded categories are planned for later phases.',
  },
  {
    q: 'Is this a substitute for medical advice?',
    a: 'No. MedVerify AI is a research and literacy tool for understanding the evidence behind a claim. Individual medical decisions should always involve a qualified clinician.',
  },
]

const COMPARISON = [
  { feature: 'Evidence retrieval (live + static)', classifier: false, llm: false, hybrid: true, medverify: true },
  { feature: 'Source-reliability weighting', classifier: false, llm: false, hybrid: false, medverify: true },
  { feature: 'Scientific consensus tally', classifier: false, llm: false, hybrid: 'partial', medverify: true },
  { feature: 'Faithfulness / hallucination check', classifier: false, llm: false, hybrid: false, medverify: true },
  { feature: 'Sentence-to-source citation mapping', classifier: false, llm: false, hybrid: 'partial', medverify: true },
  { feature: 'Evaluated against published benchmarks', classifier: true, llm: 'partial', hybrid: 'partial', medverify: true },
]

function ComparisonCell({ value }: { value: boolean | 'partial' }) {
  if (value === true) return <Check className="mx-auto h-4 w-4 text-verdict-support" />
  if (value === 'partial') return <Minus className="mx-auto h-4 w-4 text-verdict-warn" />
  return <span className="mx-auto block h-1 w-3 rounded-full bg-mist-300" />
}

export default function Landing() {
  const navigate = useNavigate()
  const [openFaq, setOpenFaq] = useState<number | null>(0)

  return (
    <div className="bg-mist-50">
      <PublicNav />

      {/* HERO */}
      <section className="relative overflow-hidden border-b border-mist-200">
        <div className="absolute inset-0 bg-grid-faint bg-[size:44px_44px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,black,transparent)]" />
        <div className="relative mx-auto max-w-7xl px-6 pb-20 pt-16 sm:pt-24">
          <div className="grid gap-16 lg:grid-cols-2 lg:items-center">
            <div>
              <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
                <Badge tone="clinical" className="mb-6">
                  <Sparkles className="h-3 w-3" /> Evidence-grounded verification, not a chatbot verdict
                </Badge>
                <h1 className="text-4xl font-semibold leading-[1.08] tracking-tight text-ink-950 sm:text-6xl">
                  Don&rsquo;t just tell me
                  <br />
                  <span className="text-clinical-600">true or false.</span>
                  <br />
                  Show me why.
                </h1>
                <p className="mt-6 max-w-lg text-lg leading-relaxed text-ink-600">
                  MedVerify AI retrieves trusted medical evidence, ranks its reliability, checks
                  scientific consensus, and verifies that its own explanation is faithful to that
                  evidence — before it ever shows you a score.
                </p>
                <div className="mt-8 flex flex-wrap items-center gap-3">
                  <Button onClick={() => navigate('/app/verify')} className="h-12 px-6 text-[15px]">
                    Verify a medical claim <ArrowRight className="h-4 w-4" />
                  </Button>
                  <Button variant="secondary" onClick={() => navigate('/app/dashboard')} className="h-12 px-6 text-[15px]">
                    Explore the dashboard
                  </Button>
                </div>
                <div className="mt-10 flex flex-wrap items-center gap-x-8 gap-y-3 text-sm text-ink-500">
                  <span className="flex items-center gap-2"><Database className="h-4 w-4 text-clinical-600" /> WHO · CDC · PubMed · PMC</span>
                  <span className="flex items-center gap-2"><GitBranch className="h-4 w-4 text-clinical-600" /> Open-weight models (Qwen 3 / Llama 3.1)</span>
                  <span className="flex items-center gap-2"><BookOpenCheck className="h-4 w-4 text-clinical-600" /> Benchmarked on PubHealth &amp; SciFact</span>
                </div>
              </motion.div>
            </div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.15 }}
            >
              <Card className="p-6 shadow-card-lg">
                <div className="mb-5 flex items-center justify-between">
                  <div>
                    <Eyebrow>Live verification</Eyebrow>
                    <p className="mt-1 text-sm font-medium text-ink-900">
                      &ldquo;Statins reduce risk of recurrent heart attacks&rdquo;
                    </p>
                  </div>
                  <Badge tone="clinical">Cardiovascular</Badge>
                </div>
                <PipelineVisualizer stages={demoStages} compact />
                <div className="mt-4 flex items-center justify-between rounded-xl bg-mist-50 p-4">
                  <div>
                    <p className="mono-label">Provisional verdict</p>
                    <p className="mt-1 text-lg font-semibold text-verdict-support">Supported</p>
                  </div>
                  <CredibilityGauge score={96} size={72} label="" />
                </div>
              </Card>
            </motion.div>
          </div>
        </div>
      </section>

      {/* PIPELINE */}
      <section id="pipeline" className="border-b border-mist-200 bg-white py-24">
        <div className="mx-auto max-w-7xl px-6">
          <SectionHeading
            eyebrow="The verification pipeline"
            title="Nine stages, one evidence thread"
            description="Every claim moves through the same auditable pipeline — from understanding the claim to a final, evidence-backed verdict. Nothing about the reasoning is hidden from you."
          />
          <div className="mt-12 grid gap-12 lg:grid-cols-[1fr_1.1fr]">
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {PIPELINE_TEMPLATE.map((stage, i) => (
                <div key={stage.id} className="rounded-xl border border-mist-200 p-4">
                  <span className="mono-label text-clinical-600">Stage {String(i + 1).padStart(2, '0')}</span>
                  <p className="mt-1.5 text-sm font-semibold text-ink-950">{stage.label}</p>
                  <p className="mt-1 text-xs leading-relaxed text-ink-500">{stage.description}</p>
                </div>
              ))}
            </div>
            <Card className="p-6 lg:p-8">
              <PipelineVisualizer stages={PIPELINE_TEMPLATE.map((s) => ({ ...s, status: 'complete' }))} />
            </Card>
          </div>
        </div>
      </section>

      {/* EVIDENCE / ARCHITECTURE */}
      <section id="evidence" className="border-b border-mist-200 py-24">
        <div className="mx-auto max-w-7xl px-6">
          <SectionHeading
            eyebrow="Evidence system"
            title="Reliability-weighted, not popularity-weighted"
            description="Every source is scored by study design and provenance — a systematic review from WHO does not carry the same weight as a preprint or a blog post."
          />
          <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[
              { label: 'WHO Guideline', score: 98, icon: ShieldCheck },
              { label: 'Systematic Review', score: 95, icon: FlaskConical },
              { label: 'RCT', score: 90, icon: ScaleIcon },
              { label: 'Preprint', score: 34, icon: Gauge },
            ].map((s) => (
              <Card key={s.label} className="p-5 text-center">
                <s.icon className="mx-auto h-5 w-5 text-clinical-600" />
                <p className="mt-3 text-2xl font-semibold tabular-nums text-ink-950">{s.score}</p>
                <p className="mt-1 text-sm text-ink-600">{s.label}</p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* RESEARCH / COMPARISON */}
      <section id="research" className="border-b border-mist-200 bg-white py-24">
        <div className="mx-auto max-w-7xl px-6">
          <SectionHeading
            eyebrow="Research contribution"
            title="What sets this apart from prior approaches"
            description="Existing verification approaches fall into three families. MedVerify AI is the first in this line of work to combine evidence grounding, reliability weighting, and mandatory faithfulness verification in one pipeline."
          />
          <div className="mt-10 overflow-x-auto rounded-2xl border border-mist-200">
            <table className="w-full min-w-[720px] border-collapse text-sm">
              <thead>
                <tr className="border-b border-mist-200 bg-mist-50 text-left">
                  <th className="px-5 py-3.5 font-medium text-ink-600">Capability</th>
                  <th className="px-5 py-3.5 text-center font-medium text-ink-600">Supervised classifier</th>
                  <th className="px-5 py-3.5 text-center font-medium text-ink-600">Pure LLM prompting</th>
                  <th className="px-5 py-3.5 text-center font-medium text-ink-600">Hybrid RAG (generic)</th>
                  <th className="px-5 py-3.5 text-center font-medium text-clinical-700">MedVerify AI</th>
                </tr>
              </thead>
              <tbody>
                {COMPARISON.map((row) => (
                  <tr key={row.feature} className="border-b border-mist-100 last:border-0">
                    <td className="px-5 py-3.5 text-ink-800">{row.feature}</td>
                    <td className="px-5 py-3.5"><ComparisonCell value={row.classifier as any} /></td>
                    <td className="px-5 py-3.5"><ComparisonCell value={row.llm as any} /></td>
                    <td className="px-5 py-3.5"><ComparisonCell value={row.hybrid as any} /></td>
                    <td className="bg-clinical-50/60 px-5 py-3.5"><ComparisonCell value={row.medverify as any} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* TEAM */}
      <section className="border-b border-mist-200 py-24">
        <div className="mx-auto max-w-7xl px-6">
          <SectionHeading eyebrow="Team" title="Built by researchers and engineers" />
          <div className="mt-10 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            {[
              { name: 'Dr. Meera Iyer', role: 'Principal Investigator, Clinical NLP' },
              { name: 'Aarav Shah', role: 'ML Engineer, Retrieval &amp; Ranking' },
              { name: 'Priya Nambiar', role: 'Research Engineer, Faithfulness Verification' },
              { name: 'Daniel Osei', role: 'Frontend &amp; Platform Engineer' },
            ].map((m) => (
              <Card key={m.name} className="p-5">
                <div className="h-11 w-11 rounded-full bg-gradient-to-br from-clinical-400 to-trust-600" />
                <p className="mt-4 text-sm font-semibold text-ink-950">{m.name}</p>
                <p className="mt-0.5 text-sm text-ink-500" dangerouslySetInnerHTML={{ __html: m.role }} />
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* FAQ */}
      <section id="faq" className="bg-white py-24">
        <div className="mx-auto max-w-3xl px-6">
          <SectionHeading eyebrow="FAQ" title="Common questions" className="mx-auto text-center" />
          <div className="mt-10 space-y-3">
            {FAQ.map((item, i) => (
              <div key={item.q} className="rounded-xl border border-mist-200">
                <button
                  onClick={() => setOpenFaq(openFaq === i ? null : i)}
                  className="flex w-full items-center justify-between px-5 py-4 text-left"
                >
                  <span className="text-sm font-medium text-ink-900">{item.q}</span>
                  <span className="text-ink-400">{openFaq === i ? '−' : '+'}</span>
                </button>
                {openFaq === i && (
                  <p className="px-5 pb-4 text-sm leading-relaxed text-ink-600">{item.a}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </div>
  )
}
