import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, BarChart, Bar } from 'recharts'
import { Eyebrow, Card } from '@/components/ui/Primitives'
import { VERIFICATION_TREND, SOURCE_USAGE, DISEASE_DISTRIBUTION } from '@/lib/mockData'

export default function Analytics() {
  return (
    <div className="space-y-6 pb-16">
      <div>
        <Eyebrow className="text-clinical-600">Analytics</Eyebrow>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-ink-950">Platform analytics</h1>
        <p className="mt-2 text-sm text-ink-600">Trends across verifications, evidence sources, and disease categories.</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="p-6">
          <p className="mb-1 text-sm font-semibold text-ink-950">Average credibility trend</p>
          <p className="mb-4 text-xs text-ink-500">Weekly average across all verifications</p>
          <div className="h-64 w-full">
            <ResponsiveContainer>
              <LineChart data={VERIFICATION_TREND} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid vertical={false} stroke="#EEF2F6" />
                <XAxis dataKey="week" tick={{ fontSize: 12, fill: '#6B7C8C' }} axisLine={{ stroke: '#DEE6ED' }} tickLine={false} />
                <YAxis domain={[70, 100]} tick={{ fontSize: 12, fill: '#6B7C8C' }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ borderRadius: 12, border: '1px solid #DEE6ED', fontSize: 13 }} />
                <Line type="monotone" dataKey="avgCredibility" stroke="#2F6FED" strokeWidth={2.5} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="p-6">
          <p className="mb-1 text-sm font-semibold text-ink-950">Evidence source usage</p>
          <p className="mb-4 text-xs text-ink-500">Citations pulled per source, last 8 weeks</p>
          <div className="h-64 w-full">
            <ResponsiveContainer>
              <BarChart data={SOURCE_USAGE} layout="vertical" margin={{ top: 4, right: 24, left: 8, bottom: 0 }}>
                <CartesianGrid horizontal={false} stroke="#EEF2F6" />
                <XAxis type="number" tick={{ fontSize: 12, fill: '#6B7C8C' }} axisLine={false} tickLine={false} />
                <YAxis dataKey="source" type="category" width={110} tick={{ fontSize: 12, fill: '#6B7C8C' }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ borderRadius: 12, border: '1px solid #DEE6ED', fontSize: 13 }} />
                <Bar dataKey="count" fill="#1B9CB5" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card className="p-6">
        <p className="mb-1 text-sm font-semibold text-ink-950">Disease category share</p>
        <p className="mb-4 text-xs text-ink-500">Share of total verifications, Phase 1 categories</p>
        <div className="grid gap-4 sm:grid-cols-3">
          {DISEASE_DISTRIBUTION.map((d) => (
            <div key={d.name} className="rounded-xl border border-mist-200 p-5">
              <div className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: d.color }} />
                <span className="text-sm font-medium text-ink-800">{d.name}</span>
              </div>
              <p className="mt-3 text-3xl font-semibold tabular-nums text-ink-950">{d.value}%</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
