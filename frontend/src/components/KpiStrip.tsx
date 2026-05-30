import { useAppStore } from '../lib/store'
import { useCountUp } from '../hooks/useCountUp'
import { badBatch } from '../lib/fixtures'

interface KpiCardProps {
  label: string
  value: number
  format: (v: number) => string
  colorClass: string
  unit?: string
}

function KpiCard({ label, value, format, colorClass, unit }: KpiCardProps) {
  const animated = useCountUp(value)
  return (
    <div className="flex flex-col items-center min-w-[120px]">
      <div className={`font-mono text-2xl font-bold tabular-nums ${colorClass}`}>
        {format(animated)}{unit && <span className="text-sm ml-0.5 opacity-70">{unit}</span>}
      </div>
      <div className="text-white/50 text-xs font-mono mt-0.5 tracking-wide uppercase">{label}</div>
    </div>
  )
}

function yieldColor(v: number): string {
  if (v >= 0.85) return 'text-yield-green'
  if (v >= 0.70) return 'text-risk-amber'
  return 'text-risk-red'
}

export default function KpiStrip() {
  const activeBatch = useAppStore((s) => s.activeBatch) ?? badBatch

  const pctFormat = (v: number) => (v * 100).toFixed(1)
  const intFormat = (v: number) => Math.round(v).toString()
  const confFormat = (v: number) => (v * 100).toFixed(0)

  return (
    <div className="h-16 bg-[#0a0f1e] border-b border-white/10 flex items-center justify-between px-6">
      <span className="text-accent font-mono text-sm font-bold tracking-widest uppercase">
        Process Control Room
      </span>
      <div className="flex items-center gap-8">
        <KpiCard
          label="Predicted Yield"
          value={activeBatch.predicted_yield}
          format={pctFormat}
          colorClass={yieldColor(activeBatch.predicted_yield)}
          unit="%"
        />
        <div className="w-px h-8 bg-white/10" />
        <KpiCard
          label="Risk Steps"
          value={activeBatch.risk_steps_detected}
          format={intFormat}
          colorClass={activeBatch.risk_steps_detected > 5 ? 'text-risk-red' : activeBatch.risk_steps_detected > 0 ? 'text-risk-amber' : 'text-white/60'}
        />
        <div className="w-px h-8 bg-white/10" />
        <KpiCard
          label="Confidence"
          value={activeBatch.confidence}
          format={confFormat}
          colorClass={activeBatch.confidence >= 0.85 ? 'text-yield-green' : 'text-white/60'}
          unit="%"
        />
        <div className="w-px h-8 bg-white/10" />
        <KpiCard
          label="Anomalous Batches"
          value={activeBatch.anomalous_batches}
          format={intFormat}
          colorClass={activeBatch.anomalous_batches > 0 ? 'text-risk-red' : 'text-white/60'}
        />
      </div>
    </div>
  )
}
