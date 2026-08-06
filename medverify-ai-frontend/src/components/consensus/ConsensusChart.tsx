import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts'
import type { ConsensusPoint } from '@/types'

export function ConsensusTimeline({ data }: { data: ConsensusPoint[] }) {
  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
          <CartesianGrid vertical={false} stroke="#EEF2F6" />
          <XAxis dataKey="year" tick={{ fontSize: 12, fill: '#6B7C8C' }} axisLine={{ stroke: '#DEE6ED' }} tickLine={false} />
          <YAxis tick={{ fontSize: 12, fill: '#6B7C8C' }} axisLine={false} tickLine={false} />
          <Tooltip
            contentStyle={{ borderRadius: 12, border: '1px solid #DEE6ED', fontSize: 13 }}
            cursor={{ fill: 'rgba(148,167,184,0.08)' }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Bar dataKey="supporting" name="Supporting" stackId="a" fill="#1E9E6B" radius={[0, 0, 0, 0]} />
          <Bar dataKey="neutral" name="Neutral" stackId="a" fill="#94A7B8" />
          <Bar dataKey="contradicting" name="Contradicting" stackId="a" fill="#C4453B" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
