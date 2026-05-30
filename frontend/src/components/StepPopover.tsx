import ReactDOM from 'react-dom'
import type { Step } from '../types/api'
import { humanize } from '../lib/featureNameMap'

interface Props {
  step: Step
  x: number
  y: number
  onClose: () => void
}

export default function StepPopover({ step, x, y }: Props) {
  const top3Shap = [...(step.shap ?? [])].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, 3)

  const content = (
    <div
      style={{
        position: 'fixed',
        left: x,
        top: y - 8,
        transform: 'translate(-50%, -100%)',
        zIndex: 9999,
        pointerEvents: 'none',
      }}
      className="bg-[#111827] border border-white/20 rounded-lg p-3 shadow-xl min-w-[220px] max-w-[280px]"
    >
      <div className="text-accent font-mono text-xs font-bold mb-2">{step.step_name}</div>
      <div className="text-white/50 text-xs font-mono mb-1">
        CI: [{step.confidence_lo.toFixed(3)}, {step.confidence_hi.toFixed(3)}]
      </div>
      {step.sensors && Object.entries(step.sensors).slice(0, 2).map(([k, v]) => (
        <div key={k} className="text-white/60 text-xs font-mono">
          {humanize(k)}: <span className="text-white">{typeof v === 'number' ? v.toFixed(2) : v}</span>
        </div>
      ))}
      {top3Shap.length > 0 && (
        <div className="mt-2 border-t border-white/10 pt-2">
          <div className="text-white/40 text-xs mb-1">SHAP Top 3</div>
          {top3Shap.map((s) => (
            <div key={s.feature} className="flex justify-between text-xs font-mono">
              <span className="text-white/70 truncate">{humanize(s.feature)}</span>
              <span className={s.contribution > 0 ? 'text-yield-green ml-2' : 'text-risk-red ml-2'}>
                {s.contribution > 0 ? '+' : ''}{s.contribution.toFixed(3)}
              </span>
            </div>
          ))}
        </div>
      )}
      <div className="mt-2 text-xs font-mono">
        Risk: <span className={step.risk_score >= 0.85 ? 'text-risk-red' : step.risk_score >= 0.70 ? 'text-risk-amber' : 'text-white/50'}>
          {step.risk_score.toFixed(3)}
        </span>
      </div>
    </div>
  )

  return ReactDOM.createPortal(content, document.body)
}
