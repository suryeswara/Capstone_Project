import { useState } from 'react'
import { Eyebrow, Card, Badge } from '@/components/ui/Primitives'
import { ConsensusTimeline } from '@/components/consensus/ConsensusChart'
import { useAppStore } from '@/store/useAppStore'
import { verdictTone } from '@/lib/utils'

export default function Consensus() {
  const history = useAppStore((s) => s.history)
  const [selected, setSelected] = useState(history[0]?.id)
  const record = history.find((v) => v.id === selected) ?? history[0]

  const totals = record.consensusTimeline.reduce(
    (acc, p) => ({
      supporting: acc.supporting + p.supporting,
      contradicting: acc.contradicting + p.contradicting,
      neutral: acc.neutral + p.neutral,
    }),
    { supporting: 0, contradicting: 0, neutral: 0 }
  )
  const total = totals.supporting + totals.contradicting + totals.neutral || 1

  return (
    <div className="space-y-6 pb-16">
      <div>
        <Eyebrow className="text-clinical-600">Scientific consensus</Eyebrow>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-ink-950">Consensus visualization</h1>
        <p className="mt-2 max-w-2xl text-sm text-ink-600">
          Rather than a single verdict, MedVerify AI tallies how many studies support, contradict,
          or remain neutral on a claim — the same way a systematic review is read.
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

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="p-6 lg:col-span-2">
          <p className="mb-1 text-sm font-semibold text-ink-950">Evidence distribution over time</p>
          <p className="mb-4 text-xs text-ink-500">{record.claim}</p>
          <ConsensusTimeline data={record.consensusTimeline} />
        </Card>

        <Card className="p-6">
          <p className="mb-4 text-sm font-semibold text-ink-950">Overall consensus split</p>
          <div className="space-y-4">
            {[
              { label: 'Supporting studies', value: totals.supporting, color: 'bg-verdict-support' },
              { label: 'Contradicting studies', value: totals.contradicting, color: 'bg-verdict-contradict' },
              { label: 'Neutral studies', value: totals.neutral, color: 'bg-verdict-neutral' },
            ].map((row) => (
              <div key={row.label}>
                <div className="mb-1.5 flex items-center justify-between text-sm">
                  <span className="text-ink-600">{row.label}</span>
                  <span className="font-medium tabular-nums text-ink-900">{row.value}</span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-mist-100">
                  <div className={`h-full ${row.color}`} style={{ width: `${(row.value / total) * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
          <div className="mt-6 rounded-xl bg-mist-50 p-4">
            <p className="mono-label mb-1">Resulting verdict</p>
            <p className={`text-lg font-semibold ${verdictTone(record.verdict).text}`}>{record.verdict}</p>
          </div>
        </Card>
      </div>
    </div>
  )
}
