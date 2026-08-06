import { Database, Cpu, Users, Activity, RefreshCcw, ShieldAlert } from 'lucide-react'
import { Eyebrow, Card, Badge, Button } from '@/components/ui/Primitives'

const SYSTEM_HEALTH = [
  { name: 'FAISS vector store', status: 'Healthy', detail: '412,884 vectors indexed · synced 6m ago' },
  { name: 'PubMed live API', status: 'Healthy', detail: 'p95 latency 1.4s · 99.8% uptime (7d)' },
  { name: 'Claim extraction (Qwen 3 8B)', status: 'Healthy', detail: 'GPU utilization 62%' },
  { name: 'Faithfulness NLI service', status: 'Degraded', detail: 'Elevated latency on batch >20 sentences' },
]

const KNOWLEDGE_BASE = [
  { category: 'Vaccination', docs: 2140, lastUpdate: '2 days ago' },
  { category: 'Cardiovascular Disease', docs: 3012, lastUpdate: '5 hours ago' },
  { category: 'Diabetes', docs: 1876, lastUpdate: '1 day ago' },
]

export default function Admin() {
  return (
    <div className="space-y-6 pb-16">
      <div className="flex items-center justify-between">
        <div>
          <Eyebrow className="text-clinical-600">Admin panel</Eyebrow>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight text-ink-950">System &amp; knowledge base</h1>
        </div>
        <Button variant="secondary"><RefreshCcw className="h-4 w-4" /> Sync knowledge base</Button>
      </div>

      <Card className="p-6">
        <div className="mb-4 flex items-center gap-2">
          <Activity className="h-4 w-4 text-clinical-600" />
          <p className="text-sm font-semibold text-ink-950">System health</p>
        </div>
        <div className="divide-y divide-mist-100">
          {SYSTEM_HEALTH.map((s) => (
            <div key={s.name} className="flex items-center justify-between py-3.5">
              <div>
                <p className="text-sm font-medium text-ink-900">{s.name}</p>
                <p className="text-xs text-ink-500">{s.detail}</p>
              </div>
              <Badge tone={s.status === 'Healthy' ? 'support' : 'warn'}>{s.status}</Badge>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="p-6">
          <div className="mb-4 flex items-center gap-2">
            <Database className="h-4 w-4 text-clinical-600" />
            <p className="text-sm font-semibold text-ink-950">Knowledge base by category</p>
          </div>
          <div className="divide-y divide-mist-100">
            {KNOWLEDGE_BASE.map((k) => (
              <div key={k.category} className="flex items-center justify-between py-3">
                <span className="text-sm text-ink-800">{k.category}</span>
                <span className="text-xs text-ink-500">{k.docs.toLocaleString()} docs · updated {k.lastUpdate}</span>
              </div>
            ))}
          </div>
        </Card>

        <Card className="p-6">
          <div className="mb-4 flex items-center gap-2">
            <ShieldAlert className="h-4 w-4 text-clinical-600" />
            <p className="text-sm font-semibold text-ink-950">Vector store integrity monitoring</p>
          </div>
          <p className="text-sm leading-relaxed text-ink-600">
            New embeddings are checked for anomalies relative to the verified-fact cluster before
            being written to the index, guarding against data-poisoning of the retrieval corpus.
          </p>
          <div className="mt-4 rounded-lg bg-verdict-support/10 p-3 text-sm text-verdict-support">
            No anomalies detected in the last sync cycle.
          </div>
        </Card>
      </div>

      <Card className="p-6">
        <div className="mb-4 flex items-center gap-2">
          <Users className="h-4 w-4 text-clinical-600" />
          <p className="text-sm font-semibold text-ink-950">User &amp; role management</p>
        </div>
        <p className="text-sm text-ink-500">Account management is scoped to Phase 2 per the implementation roadmap.</p>
      </Card>

      <Card className="p-6">
        <div className="mb-4 flex items-center gap-2">
          <Cpu className="h-4 w-4 text-clinical-600" />
          <p className="text-sm font-semibold text-ink-950">Model &amp; version provenance</p>
        </div>
        <p className="text-sm text-ink-600">
          Current production models: <span className="font-mono text-xs">qwen3-8b-instruct</span>,{' '}
          <span className="font-mono text-xs">biobert-disease-cls-v3</span>,{' '}
          <span className="font-mono text-xs">faithfulness-nli-v1.2</span>. Every stored verification
          records the exact versions used so past verdicts remain explainable after model upgrades.
        </p>
      </Card>
    </div>
  )
}
