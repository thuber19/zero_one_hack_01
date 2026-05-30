import { useState } from 'react'
import { useAppStore } from '../lib/store'
import { useCountUp } from '../hooks/useCountUp'
import { badBatch } from '../lib/fixtures'

const TOOLTIPS: Record<string, string> = {
  'Quality Score': 'How likely this batch is to produce working chips',
  'Issues Found': 'Steps where the AI detected unusual behavior',
  'Confidence': 'How certain the model is about its prediction',
  'Flagged Batches': 'Batches that may contain defects',
}

interface KpiCardProps {
  label: string
  value: number
  format: (v: number) => string
  colorClass: string
  unit?: string
}

function KpiCard({ label, value, format, colorClass, unit }: KpiCardProps) {
  const animated = useCountUp(value)
  const [showTip, setShowTip] = useState(false)
  const tip = TOOLTIPS[label]

  return (
    <div className="flex flex-col items-center min-w-[130px] relative">
      <div className={`font-mono text-2xl font-bold tabular-nums ${colorClass}`}>
        {format(animated)}{unit && <span className="text-sm ml-0.5 opacity-70">{unit}</span>}
      </div>
      <div className="flex items-center gap-1">
        <div className="text-white/50 text-xs font-mono mt-0.5 tracking-wide uppercase">{label}</div>
        {tip && (
          <div
            className="relative"
            onMouseEnter={() => setShowTip(true)}
            onMouseLeave={() => setShowTip(false)}
          >
            <span className="text-white/20 text-xs cursor-help hover:text-white/50 transition-colors">(?)</span>
            {showTip && (
              <div
                className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 rounded-lg text-xs font-mono text-white/80 whitespace-nowrap z-50 pointer-events-none"
                style={{ backgroundColor: '#1a2540', border: '1px solid #ffffff20', boxShadow: '0 4px 16px rgba(0,0,0,0.4)' }}
              >
                {tip}
                <div className="absolute top-full left-1/2 -translate-x-1/2 w-0 h-0" style={{ borderLeft: '5px solid transparent', borderRight: '5px solid transparent', borderTop: '5px solid #1a2540' }} />
              </div>
            )}
          </div>
        )}
      </div>
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
        Fab Process Monitor
      </span>
      <div className="flex items-center gap-8">
        <KpiCard
          label="Quality Score"
          value={activeBatch.predicted_yield}
          format={pctFormat}
          colorClass={yieldColor(activeBatch.predicted_yield)}
          unit="%"
        />
        <div className="w-px h-8 bg-white/10" />
        <KpiCard
          label="Issues Found"
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
          label="Flagged Batches"
          value={activeBatch.anomalous_batches}
          format={intFormat}
          colorClass={activeBatch.anomalous_batches > 0 ? 'text-risk-red' : 'text-white/60'}
        />
      </div>
    </div>
  )
}
