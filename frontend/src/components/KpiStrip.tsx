import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getEvalMetrics } from '../lib/api'
import { useCountUp } from '../hooks/useCountUp'
import type { EvalMetrics } from '../types/api'

const TOOLTIPS: Record<string, string> = {
  'Task 1 Top-1': 'Fraction where the top predicted next step exactly matches ground truth',
  'Task 2 Block Acc': 'Coarse-grained accuracy over 9 process blocks (LITHO, ETCH, DOPING...)',
  'Rule Validity': '100% of completed sequences pass all 10 physics/process rules',
  'Task 3 Hybrid': 'Physics-hybrid anomaly detection binary accuracy on validation set',
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

function PlaceholderCard({ label }: { label: string }) {
  return (
    <div className="flex flex-col items-center min-w-[130px] gap-1.5">
      <div className="h-7 w-20 rounded bg-white/5 animate-pulse" />
      <div className="text-white/30 text-xs font-mono tracking-wide uppercase">{label}</div>
    </div>
  )
}

const PLACEHOLDER_LABELS = ['Task 1 Top-1', 'Task 2 Block Acc', 'Rule Validity', 'Task 3 Hybrid']

export default function KpiStrip() {
  const { data: metrics, isLoading, isError } = useQuery<EvalMetrics>({
    queryKey: ['evalMetrics'],
    queryFn: getEvalMetrics,
    staleTime: Infinity,
  })

  const showPlaceholders = isLoading || isError || !metrics

  return (
    <div className="h-16 bg-[#0a0f1e] border-b border-white/10 flex items-center justify-between px-6">
      <span className="text-accent font-mono text-sm font-bold tracking-widest uppercase">
        Team TBD: ProcSeq Monitor
      </span>
      <div className="flex items-center gap-8">
        {showPlaceholders ? (
          <>
            {PLACEHOLDER_LABELS.map((label, i) => (
              <>
                {i > 0 && <div key={`sep-${i}`} className="w-px h-8 bg-white/10" />}
                <PlaceholderCard key={label} label={label} />
              </>
            ))}
          </>
        ) : (
          <>
            <KpiCard
              label="Task 1 Top-1"
              value={metrics.task1_nextstep.top1 * 100}
              format={(v) => v.toFixed(1)}
              colorClass="text-yield-green"
              unit="%"
            />
            <div className="w-px h-8 bg-white/10" />
            <KpiCard
              label="Task 2 Block Acc"
              value={metrics.task2_completion.block_accuracy * 100}
              format={(v) => v.toFixed(1)}
              colorClass="text-yield-green"
              unit="%"
            />
            <div className="w-px h-8 bg-white/10" />
            <KpiCard
              label="Rule Validity"
              value={metrics.task2_logic_validity * 100}
              format={(v) => v.toFixed(0)}
              colorClass="text-yield-green"
              unit="%"
            />
            <div className="w-px h-8 bg-white/10" />
            <KpiCard
              label="Task 3 Hybrid"
              value={metrics.task3_hybrid.binary_accuracy * 100}
              format={(v) => v.toFixed(0)}
              colorClass="text-yield-green"
              unit="%"
            />
          </>
        )}
      </div>
    </div>
  )
}
