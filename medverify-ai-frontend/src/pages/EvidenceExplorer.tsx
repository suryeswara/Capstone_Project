import { useMemo, useState } from 'react'
import { Search } from 'lucide-react'
import { Eyebrow, Badge } from '@/components/ui/Primitives'
import { EvidenceCard } from '@/components/evidence/EvidenceCard'
import { useAppStore } from '@/store/useAppStore'
import type { EvidenceItem, SourceType } from '@/types'

const SOURCE_FILTERS: SourceType[] = ['WHO Guideline', 'CDC Guideline', 'Systematic Review', 'RCT', 'PubMed Article', 'Cohort Study', 'Preprint']

export default function EvidenceExplorer() {
  const history = useAppStore((s) => s.history)
  const [query, setQuery] = useState('')
  const [sourceFilter, setSourceFilter] = useState<SourceType | 'all'>('all')

  const allEvidence: EvidenceItem[] = useMemo(() => {
    const map = new Map<string, EvidenceItem>()
    history.forEach((v) => v.evidence.forEach((e) => map.set(e.id, e)))
    return Array.from(map.values()).sort((a, b) => b.reliabilityScore - a.reliabilityScore)
  }, [history])

  const filtered = allEvidence.filter((e) => {
    const matchesQuery = query.trim()
      ? (e.title + e.abstract + e.authors).toLowerCase().includes(query.toLowerCase())
      : true
    const matchesSource = sourceFilter === 'all' ? true : e.source === sourceFilter
    return matchesQuery && matchesSource
  })

  return (
    <div className="space-y-6 pb-16">
      <div>
        <Eyebrow className="text-clinical-600">Evidence explorer</Eyebrow>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-ink-950">Browse the evidence corpus</h1>
        <p className="mt-2 max-w-2xl text-sm text-ink-600">
          Every piece of evidence retrieved across all verifications, ranked by reliability score.
          Sourced from WHO, CDC, PubMed, and PubMed Central.
        </p>
      </div>

      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <div className="flex flex-1 items-center gap-2 rounded-xl border border-mist-200 bg-white px-3 py-2.5">
          <Search className="h-4 w-4 text-ink-400" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search titles, authors, abstracts…"
            className="w-full bg-transparent text-sm text-ink-800 placeholder:text-ink-400 focus:outline-none"
          />
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => setSourceFilter('all')}
          className={sourceFilter === 'all' ? 'opacity-100' : 'opacity-60 hover:opacity-100'}
        >
          <Badge tone={sourceFilter === 'all' ? 'clinical' : 'neutral'}>All sources</Badge>
        </button>
        {SOURCE_FILTERS.map((s) => (
          <button
            key={s}
            onClick={() => setSourceFilter(s)}
            className={sourceFilter === s ? 'opacity-100' : 'opacity-60 hover:opacity-100'}
          >
            <Badge tone={sourceFilter === s ? 'clinical' : 'neutral'}>{s}</Badge>
          </button>
        ))}
      </div>

      <p className="text-xs text-ink-500">{filtered.length} sources</p>

      <div className="grid gap-4 lg:grid-cols-2">
        {filtered.map((e) => (
          <EvidenceCard key={e.id} evidence={e} />
        ))}
        {filtered.length === 0 && (
          <div className="col-span-2 rounded-xl border border-dashed border-mist-300 p-10 text-center text-sm text-ink-500">
            No evidence matches your filters yet. Try a different search term or source type.
          </div>
        )}
      </div>
    </div>
  )
}
