import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Sparkles, Send, FlaskConical } from 'lucide-react'
import { Card, Button, Badge, Eyebrow } from '@/components/ui/Primitives'
import { PipelineVisualizer } from '@/components/pipeline/PipelineVisualizer'
import { CredibilityGauge } from '@/components/credibility/CredibilityGauge'
import { useAppStore } from '@/store/useAppStore'
import { EXAMPLE_CLAIMS } from '@/lib/mockData'
import { verdictTone } from '@/lib/utils'

export default function Verify() {
  const navigate = useNavigate()
  const [text, setText] = useState('')
  const { runVerification, isRunning, stages, result, reset } = useAppStore()

  const submit = (claim: string) => {
    if (!claim.trim()) return
    setText(claim)
    runVerification(claim)
  }

  return (
    <div className="mx-auto max-w-4xl space-y-8 pb-16">
      <div>
        <Eyebrow className="text-clinical-600">Claim verification workspace</Eyebrow>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-ink-950">Verify a medical claim</h1>
        <p className="mt-2 text-sm text-ink-600">
          Describe a health claim in plain language. MedVerify AI will retrieve evidence, check
          consensus, and verify its own explanation before showing a verdict.
        </p>
      </div>

      {!result && !isRunning && (
        <Card className="p-6">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={4}
            placeholder="e.g. Statins reduce the risk of recurrent heart attacks in people with existing heart disease."
            className="w-full resize-none rounded-xl border border-mist-200 bg-mist-50 p-4 text-[15px] text-ink-900 placeholder:text-ink-400 focus:border-clinical-400 focus:outline-none focus:ring-2 focus:ring-clinical-400/25"
          />
          <div className="mt-4 flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs text-ink-500">
              <Sparkles className="h-3.5 w-3.5 text-clinical-500" />
              AI assistant will classify disease category automatically
            </div>
            <Button onClick={() => submit(text)} disabled={!text.trim()}>
              Verify claim <Send className="h-4 w-4" />
            </Button>
          </div>

          <div className="mt-6 border-t border-mist-100 pt-5">
            <p className="mono-label mb-3">Try an example</p>
            <div className="flex flex-wrap gap-2">
              {EXAMPLE_CLAIMS.map((ex) => (
                <button
                  key={ex.text}
                  onClick={() => submit(ex.text)}
                  className="rounded-full border border-mist-200 bg-white px-3.5 py-1.5 text-xs font-medium text-ink-600 hover:border-clinical-400 hover:text-clinical-700"
                >
                  {ex.text}
                </button>
              ))}
            </div>
          </div>
        </Card>
      )}

      <AnimatePresence>
        {(isRunning || result) && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <Card className="p-6">
              <div className="mb-5 flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <p className="mono-label mb-1">Claim under verification</p>
                  <p className="text-[15px] font-medium text-ink-900">{text}</p>
                </div>
                {isRunning && <Badge tone="clinical">Processing</Badge>}
              </div>
              <PipelineVisualizer stages={stages} />
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
            className="space-y-4"
          >
            <Card className="flex flex-col items-center gap-6 p-8 sm:flex-row sm:justify-between">
              <div className="text-center sm:text-left">
                <p className="mono-label mb-2">Evidence-backed final verdict</p>
                <p className={`text-3xl font-semibold ${verdictTone(result.verdict).text}`}>{result.verdict}</p>
                <p className="mt-2 max-w-md text-sm text-ink-600">
                  Based on {result.evidence.length} retrieved sources across {result.disease.toLowerCase()} literature.
                </p>
                <div className="mt-5 flex flex-wrap justify-center gap-2 sm:justify-start">
                  <Button onClick={() => navigate(`/app/report/${result.id}`)}>
                    <FlaskConical className="h-4 w-4" /> View full report
                  </Button>
                  <Button variant="secondary" onClick={() => { reset(); setText('') }}>
                    Verify another claim
                  </Button>
                </div>
              </div>
              <CredibilityGauge score={result.credibility.overall} label="Overall credibility" />
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
