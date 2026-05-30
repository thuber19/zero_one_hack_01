import ReactDOM from 'react-dom'
import type { Step } from '../types/api'
import { stepDescription } from '../lib/stepDescriptions'

interface Props {
  step: Step
  x: number
  y: number
  onClose: () => void
}

export default function StepPopover({ step, x, y }: Props) {
  const desc = stepDescription(step.step_name)
  const riskPct = Math.round(step.risk_score * 100)
  const isHighRisk = step.risk_score >= 0.85
  const isMedRisk = step.risk_score >= 0.70

  const content = (
    <div
      style={{
        position: 'fixed',
        left: x,
        top: y - 12,
        transform: 'translate(-50%, -100%)',
        zIndex: 9999,
        pointerEvents: 'none',
      }}
      className="bg-[#0d1830] border border-white/20 rounded-xl p-3 shadow-2xl min-w-[240px] max-w-[320px]"
    >
      {/* Step name */}
      <div className="font-mono text-sm font-bold text-white mb-1">{step.step_name}</div>

      {/* Real description from CSV */}
      {desc && (
        <div className="text-white/60 text-xs leading-relaxed mb-2">{desc}</div>
      )}

      {/* Risk score */}
      <div className="flex items-center gap-2 mt-1">
        <div
          className="text-xs font-mono font-bold px-2 py-0.5 rounded"
          style={{
            backgroundColor: isHighRisk ? '#ff444422' : isMedRisk ? '#ffaa0022' : '#1e2a3a',
            color: isHighRisk ? '#ff4444' : isMedRisk ? '#ffaa00' : '#ffffff60',
            border: `1px solid ${isHighRisk ? '#ff444440' : isMedRisk ? '#ffaa0040' : 'transparent'}`,
          }}
        >
          {riskPct}% anomaly score
        </div>
        {(isHighRisk || isMedRisk) && (
          <span className="text-xs font-mono text-white/30">
            ±{Math.round((step.confidence_hi - step.confidence_lo) * 50)}%
          </span>
        )}
      </div>

      {/* Arrow */}
      <div
        style={{
          position: 'absolute',
          bottom: -5,
          left: '50%',
          transform: 'translateX(-50%)',
          width: 10,
          height: 10,
          backgroundColor: '#0d1830',
          border: '1px solid rgba(255,255,255,0.2)',
          clipPath: 'polygon(0 0, 100% 0, 50% 100%)',
        }}
      />
    </div>
  )

  return ReactDOM.createPortal(content, document.body)
}
