import { Link } from 'react-router-dom'
import { ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts'
import { ArrowRight, ShieldCheck, FileCheck2, AlertTriangle, TrendingUp } from 'lucide-react'
import { Card, Badge, Button } from '@/components/ui/Primitives'
import { useAppStore } from '@/store/useAppStore'
import { DISEASE_DISTRIBUTION, VERIFICATION_TREND } from '@/lib/mockData'
import { formatDate, verdictTone } from '@/lib/utils'

const STATS = [
  { label: 'Verifications this month', value: '1,284', icon: FileCheck2, delta: '+18% vs last month' },
  { label: 'Avg. credibility score', value: '87.4', icon: ShieldCheck, delta: '+2.1 pts' },
  { label: 'Explanation flags raised', value: '63', icon: AlertTriangle, delta: '4.9% of runs' },
  { label: 'Consensus agreement', value: '94%', icon: TrendingUp, delta: 'vs. expert review panel' },
]

export default function Dashboard() {
  const history = useAppStore((s) => s.history)

  return (
    <div className="space-y-8 pb-16">
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
        <div>
          <p className="mono-label text-clinical-600">Welcome back, Dr. Kapoor</p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight text-ink-950">Verification overview</h1>
        </div>
        <Link to="/app/verify">
          <Button>
            New verification <ArrowRight className="h-4 w-4" />
          </Button>
        </Link>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {STATS.map((s) => (
          <Card key={s.label} className="p-5">
            <div className="flex items-center justify-between">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-clinical-50 text-clinical-600">
                <s.icon className="h-4.5 w-4.5" />
              </div>
            </div>
            <p className="mt-4 text-2xl font-semibold tabular-nums text-ink-950">{s.value}</p>
            <p className="mt-1 text-sm text-ink-500">{s.label}</p>
            <p className="mt-2 text-xs font-medium text-verdict-support">{s.delta}</p>
          </Card>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="p-6 lg:col-span-2">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-ink-950">Verification volume &amp; avg. credibility</p>
              <p className="text-xs text-ink-500">Last 8 weeks</p>
            </div>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer>
              <AreaChart data={VERIFICATION_TREND} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="volGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#1B9CB5" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#1B9CB5" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid vertical={false} stroke="#EEF2F6" />
                <XAxis dataKey="week" tick={{ fontSize: 12, fill: '#6B7C8C' }} axisLine={{ stroke: '#DEE6ED' }} tickLine={false} />
                <YAxis tick={{ fontSize: 12, fill: '#6B7C8C' }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ borderRadius: 12, border: '1px solid #DEE6ED', fontSize: 13 }} />
                <Area type="monotone" dataKey="verifications" stroke="#1B9CB5" strokeWidth={2} fill="url(#volGrad)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="p-6">
          <p className="mb-4 text-sm font-semibold text-ink-950">Disease distribution</p>
          <div className="h-48 w-full">
            <ResponsiveContainer>
              <PieChart>
                <Pie data={DISEASE_DISTRIBUTION} dataKey="value" nameKey="name" innerRadius={50} outerRadius={72} paddingAngle={3}>
                  {DISEASE_DISTRIBUTION.map((d) => (
                    <Cell key={d.name} fill={d.color} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ borderRadius: 12, border: '1px solid #DEE6ED', fontSize: 13 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 space-y-2">
            {DISEASE_DISTRIBUTION.map((d) => (
              <div key={d.name} className="flex items-center justify-between text-sm">
                <span className="flex items-center gap-2 text-ink-600">
                  <span className="h-2 w-2 rounded-full" style={{ backgroundColor: d.color }} />
                  {d.name}
                </span>
                <span className="font-medium tabular-nums text-ink-900">{d.value}%</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card className="p-6">
        <div className="mb-4 flex items-center justify-between">
          <p className="text-sm font-semibold text-ink-950">Recent claims</p>
          <Link to="/app/history" className="text-sm font-medium text-clinical-700 hover:text-clinical-600">
            View all
          </Link>
        </div>
        <div className="divide-y divide-mist-100">
          {history.slice(0, 5).map((v) => {
            const tone = verdictTone(v.verdict)
            return (
              <Link
                key={v.id}
                to={`/app/report/${v.id}`}
                className="flex flex-col gap-2 py-4 first:pt-0 last:pb-0 sm:flex-row sm:items-center sm:justify-between"
              >
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-ink-900">{v.claim}</p>
                  <p className="mt-1 text-xs text-ink-500">{v.disease} · {formatDate(v.submittedAt)}</p>
                </div>
                <div className="flex items-center gap-3">
                  <Badge className={`${tone.bg} ${tone.text} ${tone.ring}`}>{v.verdict}</Badge>
                  <span className="text-sm font-semibold tabular-nums text-ink-900">{v.credibility.overall}</span>
                </div>
              </Link>
            )
          })}
        </div>
      </Card>
    </div>
  )
}
