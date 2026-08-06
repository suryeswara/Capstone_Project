import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Search, Bookmark, BookmarkCheck } from 'lucide-react'
import { Eyebrow, Card, Badge } from '@/components/ui/Primitives'
import { useAppStore } from '@/store/useAppStore'
import { formatDate, verdictTone } from '@/lib/utils'

const VERDICT_FILTERS = ['All', 'Supported', 'Contradicted', 'Mixed', 'Insufficient Evidence']

export default function HistoryPage() {
  const { history, toggleBookmark } = useAppStore()
  const [query, setQuery] = useState('')
  const [verdictFilter, setVerdictFilter] = useState('All')
  const [bookmarkedOnly, setBookmarkedOnly] = useState(false)

  const filtered = useMemo(() => {
    return history.filter((v) => {
      const matchesQuery = query.trim() ? v.claim.toLowerCase().includes(query.toLowerCase()) : true
      const matchesVerdict = verdictFilter === 'All' ? true : v.verdict === verdictFilter
      const matchesBookmark = bookmarkedOnly ? !!v.bookmarked : true
      return matchesQuery && matchesVerdict && matchesBookmark
    })
  }, [history, query, verdictFilter, bookmarkedOnly])

  return (
    <div className="space-y-6 pb-16">
      <div>
        <Eyebrow className="text-clinical-600">History</Eyebrow>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-ink-950">Verification history</h1>
        <p className="mt-2 text-sm text-ink-600">Search, filter, and revisit every claim you&rsquo;ve verified.</p>
      </div>

      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <div className="flex flex-1 items-center gap-2 rounded-xl border border-mist-200 bg-white px-3 py-2.5">
          <Search className="h-4 w-4 text-ink-400" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search claims…"
            className="w-full bg-transparent text-sm text-ink-800 placeholder:text-ink-400 focus:outline-none"
          />
        </div>
        <button
          onClick={() => setBookmarkedOnly((b) => !b)}
          className="flex items-center gap-2 rounded-xl border border-mist-200 bg-white px-3.5 py-2.5 text-sm text-ink-700 hover:border-clinical-400"
        >
          {bookmarkedOnly ? <BookmarkCheck className="h-4 w-4 text-clinical-600" /> : <Bookmark className="h-4 w-4" />}
          Bookmarked
        </button>
      </div>

      <div className="flex flex-wrap gap-2">
        {VERDICT_FILTERS.map((f) => (
          <button key={f} onClick={() => setVerdictFilter(f)}>
            <Badge tone={verdictFilter === f ? 'clinical' : 'neutral'}>{f}</Badge>
          </button>
        ))}
      </div>

      <Card className="divide-y divide-mist-100 p-0">
        {filtered.map((v) => {
          const tone = verdictTone(v.verdict)
          return (
            <div key={v.id} className="flex items-center gap-4 p-5">
              <Link to={`/app/report/${v.id}`} className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-ink-900 hover:text-clinical-700">{v.claim}</p>
                <p className="mt-1 text-xs text-ink-500">
                  {v.disease} · {formatDate(v.submittedAt)} · {v.evidence.length} sources
                </p>
              </Link>
              <Badge className={`${tone.bg} ${tone.text} ${tone.ring}`}>{v.verdict}</Badge>
              <span className="w-10 text-right text-sm font-semibold tabular-nums text-ink-900">{v.credibility.overall}</span>
              <button onClick={() => toggleBookmark(v.id)} className="text-ink-400 hover:text-clinical-600">
                {v.bookmarked ? <BookmarkCheck className="h-4 w-4 text-clinical-600" /> : <Bookmark className="h-4 w-4" />}
              </button>
            </div>
          )
        })}
        {filtered.length === 0 && (
          <div className="p-10 text-center text-sm text-ink-500">No verifications match these filters.</div>
        )}
      </Card>
    </div>
  )
}
