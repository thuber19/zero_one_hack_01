import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer } from 'recharts'
import { humanize } from '../lib/featureNameMap'
import type { ShapContribution } from '../types/api'

interface Props {
  shap: ShapContribution[] | null | undefined
  stepName?: string
}

export default function ShapPanel({ shap, stepName }: Props) {
  if (!shap || shap.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-white/30 font-mono text-sm">
        SHAP data not available for this step
      </div>
    )
  }

  const sorted = [...shap]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 10)
    .map((s) => ({
      ...s,
      label: humanize(s.feature),
    }))

  return (
    <div className="space-y-2" style={{ transition: 'all 300ms ease' }}>
      {stepName && (
        <div className="text-accent font-mono text-sm font-bold">{stepName}</div>
      )}
      <div className="text-white/40 text-xs font-mono mb-2">SHAP Feature Contributions</div>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart
          data={sorted}
          layout="vertical"
          margin={{ top: 4, right: 20, left: 140, bottom: 4 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
          <XAxis type="number" tick={{ fill: '#ffffff60', fontSize: 11, fontFamily: 'monospace' }} />
          <YAxis
            type="category"
            dataKey="label"
            width={135}
            tick={{ fill: '#ffffff80', fontSize: 11, fontFamily: 'monospace' }}
          />
          <Tooltip
            contentStyle={{ backgroundColor: '#111827', border: '1px solid #ffffff20', fontFamily: 'monospace', fontSize: 12 }}
            labelStyle={{ color: '#ffffff' }}
            formatter={(value: number) => [value.toFixed(4), 'Contribution']}
          />
          <Bar dataKey="contribution" radius={[0, 3, 3, 0]}>
            {sorted.map((entry, idx) => (
              <Cell key={idx} fill={entry.contribution > 0 ? '#00cc66' : '#ff4444'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
