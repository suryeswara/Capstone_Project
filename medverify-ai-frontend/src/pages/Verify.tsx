import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Sparkles, Send, FlaskConical, Image as ImageIcon, Type, Upload, FileText, CheckCircle } from 'lucide-react'
import { Card, Button, Badge, Eyebrow } from '@/components/ui/Primitives'
import { PipelineVisualizer } from '@/components/pipeline/PipelineVisualizer'
import { CredibilityGauge } from '@/components/credibility/CredibilityGauge'
import { useAppStore } from '@/store/useAppStore'
import { EXAMPLE_CLAIMS } from '@/lib/mockData'
import { verdictTone } from '@/lib/utils'

export default function Verify() {
  const navigate = useNavigate()
  const [mode, setMode] = useState<'text' | 'image'>('text')
  const [text, setText] = useState('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const { runVerification, runImageVerification, isRunning, stages, result, activeClaim, reset } = useAppStore()

  const submitText = (claim: string) => {
    if (!claim.trim()) return
    setText(claim)
    runVerification(claim)
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setImagePreview(URL.createObjectURL(file))
    }
  }

  const submitImage = () => {
    if (!selectedFile) return
    runImageVerification(selectedFile)
  }

  return (
    <div className="mx-auto max-w-4xl space-y-8 pb-16">
      <div>
        <Eyebrow className="text-clinical-600">Claim verification workspace</Eyebrow>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-ink-950">Verify a medical claim</h1>
        <p className="mt-2 text-sm text-ink-600">
          Submit a health claim via text or upload a screenshot (WhatsApp forward, Instagram post, Twitter/X, or infographic). MedVerify AI extracts, retrieves evidence, checks consensus, and evaluates explanation faithfulness.
        </p>
      </div>

      {!result && !isRunning && (
        <Card className="p-6">
          {/* Mode Switcher */}
          <div className="mb-6 flex items-center gap-2 border-b border-mist-100 pb-4">
            <button
              onClick={() => setMode('text')}
              className={`flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-medium transition-colors ${
                mode === 'text'
                  ? 'bg-clinical-50 text-clinical-700 font-semibold'
                  : 'text-ink-600 hover:bg-mist-50'
              }`}
            >
              <Type className="h-4 w-4" /> Text claim
            </button>
            <button
              onClick={() => setMode('image')}
              className={`flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-medium transition-colors ${
                mode === 'image'
                  ? 'bg-clinical-50 text-clinical-700 font-semibold'
                  : 'text-ink-600 hover:bg-mist-50'
              }`}
            >
              <ImageIcon className="h-4 w-4" /> Upload screenshot / image (OCR)
            </button>
          </div>

          {mode === 'text' ? (
            <div>
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
                  BioBERT will classify disease domain automatically
                </div>
                <Button onClick={() => submitText(text)} disabled={!text.trim()}>
                  Verify claim <Send className="h-4 w-4" />
                </Button>
              </div>

              <div className="mt-6 border-t border-mist-100 pt-5">
                <p className="mono-label mb-3">Try an example</p>
                <div className="flex flex-wrap gap-2">
                  {EXAMPLE_CLAIMS.map((ex) => (
                    <button
                      key={ex.text}
                      onClick={() => submitText(ex.text)}
                      className="rounded-full border border-mist-200 bg-white px-3.5 py-1.5 text-xs font-medium text-ink-600 hover:border-clinical-400 hover:text-clinical-700"
                    >
                      {ex.text}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileSelect}
                accept="image/*"
                className="hidden"
              />
              
              {!selectedFile ? (
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-mist-300 bg-mist-50/50 p-8 text-center transition-colors hover:border-clinical-400 hover:bg-clinical-50/30"
                >
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-clinical-100 text-clinical-700">
                    <Upload className="h-6 w-6" />
                  </div>
                  <p className="mt-3 text-sm font-medium text-ink-900">
                    Click to upload or drag & drop a screenshot
                  </p>
                  <p className="mt-1 text-xs text-ink-500">
                    Supports WhatsApp screenshots, Instagram posts, Twitter/X infographics (PNG, JPG, WEBP)
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center justify-between rounded-xl border border-mist-200 bg-white p-4">
                    <div className="flex items-center gap-3">
                      {imagePreview && (
                        <img
                          src={imagePreview}
                          alt="Claim upload preview"
                          className="h-16 w-16 rounded-lg object-cover border border-mist-200"
                        />
                      )}
                      <div>
                        <p className="text-sm font-medium text-ink-900">{selectedFile.name}</p>
                        <p className="text-xs text-ink-500">{(selectedFile.size / 1024).toFixed(1)} KB • Ready for OCR claim extraction</p>
                      </div>
                    </div>
                    <Button variant="secondary" onClick={() => { setSelectedFile(null); setImagePreview(null); }}>
                      Change image
                    </Button>

                  </div>

                  <div className="flex items-center justify-between pt-2">
                    <div className="flex items-center gap-2 text-xs text-ink-500">
                      <Sparkles className="h-3.5 w-3.5 text-clinical-500" />
                      OCR will extract text & filter social media UI noise automatically
                    </div>
                    <Button onClick={submitImage}>
                      Extract & verify claim <Send className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              )}
            </div>
          )}
        </Card>
      )}

      <AnimatePresence>
        {(isRunning || result) && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <Card className="p-6">
              <div className="mb-5 flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <p className="mono-label mb-1">Claim under verification</p>
                  <p className="text-[15px] font-medium text-ink-900">{activeClaim || text}</p>
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
                  <Button variant="secondary" onClick={() => { reset(); setText(''); setSelectedFile(null); setImagePreview(null); }}>
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
