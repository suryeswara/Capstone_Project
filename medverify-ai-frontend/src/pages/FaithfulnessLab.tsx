import { useState } from 'react'
import { Eyebrow, Card, Badge } from '@/components/ui/Primitives'
import { FaithfulnessView } from '@/components/faithfulness/FaithfulnessView'
import { useAppStore } from '@/store/useAppStore'
import { CheckCircle2, AlertTriangle, XCircle } from 'lucide-react'

export default function FaithfulnessLab() {
  const history = useAppStore((s) => s.history)
  const [selected, setSelected] = useState(history[0]?.id)
  const record = history.find((v) => v.id === selected) ?? history[0]

  const counts = record.explanation.reduce(
    (acc, s) => ({ ...acc, [s.status]: acc[s.status] + 1 }),
    { verified: 0, unsupported: 0, contradiction: 0 } as Record<string, number>
  )

  return (
    <div className="space-y-6 pb-16">
      <div>
        <Eyebrow className="text-clinical-600">Faithfulness verification</Eyebrow>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-ink-950">Faithfulness lab</h1>
        <p className="mt-2 max-w-2xl text-sm text-ink-600">
          Every AI-generated explanation sentence is checked against the retrieved evidence with an
          NLI entailment model before it is ever displayed. Select a sentence to see its evidence mapping.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {history.map((v) => (
          <button key={v.id} onClick={() => setSelected(v.id)}>
            <Badge tone={selected === v.id ? 'clinical' : 'neutral'} className="max-w-xs truncate">
              {v.claim}
            </Badge>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-4">
        <Card className="flex items-center gap-3 p-4">
          <CheckCircle2 className="h-5 w-5 text-verdict-support" />
          <div>
            <p className="text-lg font-semibold tabular-nums text-ink-950">{counts.verified}</p>
            <p className="text-xs text-ink-500">Verified sentences</p>
          </div>
        </Card>
        <Card className="flex items-center gap-3 p-4">
          <AlertTriangle className="h-5 w-5 text-verdict-warn" />
          <div>
            <p className="text-lg font-semibold tabular-nums text-ink-950">{counts.unsupported}</p>
            <p className="text-xs text-ink-500">Unsupported sentences</p>
          </div>
        </Card>
        <Card className="flex items-center gap-3 p-4">
          <XCircle className="h-5 w-5 text-verdict-contradict" />
          <div>
            <p className="text-lg font-semibold tabular-nums text-ink-950">{counts.contradiction}</p>
            <p className="text-xs text-ink-500">Contradictions</p>
          </div>
        </Card>
      </div>

      <Card className="p-6">
        <FaithfulnessView sentences={record.explanation} evidence={record.evidence} />
      </Card>
    </div>
  )
}
