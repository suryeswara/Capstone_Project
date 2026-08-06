import { useState } from 'react'
import { ChevronDown, ExternalLink, Landmark } from 'lucide-react'
import type { EvidenceItem } from '@/types'
import { Badge, Card } from '@/components/ui/Primitives'
import { cn } from '@/lib/utils'

const stanceTone: Record<EvidenceItem['stance'], 'support' | 'contradict' | 'neutral'> = {
  supporting: 'support',
  contradicting: 'contradict',
  neutral: 'neutral',
}

const stanceLabel: Record<EvidenceItem['stance'], string> = {
  supporting: 'Supports claim',
  contradicting: 'Contradicts claim',
  neutral: 'Neutral / mixed',
}

export function EvidenceCard({ evidence }: { evidence: EvidenceItem }) {
  const [open, setOpen] = useState(false)
  return (
    <Card className="p-5">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="clinical">{evidence.source}</Badge>
            <Badge tone={stanceTone[evidence.stance] === 'neutral' ? 'neutral' : stanceTone[evidence.stance]}>
              {stanceLabel[evidence.stance]}
            </Badge>
            <span className="mono-label">{evidence.year}</span>
          </div>
          <h4 className="mt-2 text-[15px] font-semibold leading-snug text-ink-950">{evidence.title}</h4>
          <p className="mt-1 text-sm text-ink-500">{evidence.authors}</p>
        </div>
        <div className="flex flex-none flex-col items-end gap-1">
          <div className="flex items-center gap-1.5">
            <Landmark className="h-3.5 w-3.5 text-clinical-600" />
            <span className="text-sm font-semibold tabular-nums text-ink-900">{evidence.reliabilityScore}</span>
          </div>
          <span className="mono-label">reliability</span>
        </div>
      </div>

      <button
        onClick={() => setOpen((o) => !o)}
        className="mt-4 flex w-full items-center justify-between rounded-lg border border-mist-200 bg-mist-50 px-3 py-2 text-left text-sm font-medium text-ink-700 hover:bg-mist-100"
      >
        {open ? 'Hide abstract' : 'Expand abstract'}
        <ChevronDown className={cn('h-4 w-4 transition-transform', open && 'rotate-180')} />
      </button>
      {open && (
        <div className="mt-3 space-y-3 border-t border-mist-100 pt-3">
          <p className="text-sm leading-relaxed text-ink-600">{evidence.abstract}</p>
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-ink-500">
            <span className="font-mono">DOI: {evidence.doi}</span>
            <a
              href={evidence.url}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 font-medium text-clinical-700 hover:text-clinical-600"
            >
              View source <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </div>
      )}
    </Card>
  )
}
