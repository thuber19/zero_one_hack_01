import { useNavigate } from 'react-router-dom'
import { useAppStore } from '../lib/store'
import { badBatch, goodBatch } from '../lib/fixtures'

const REAL_STEPS = [
  { step: 'RECEIVE WAFER LOT', desc: 'Verify lot ID and route version' },
  { step: 'THERMAL OXIDATION', desc: 'Grow silicon oxide at ~950°C' },
  { step: 'SPIN COAT PHOTORESIST', desc: 'Apply light-sensitive coating' },
  { step: 'EXPOSE LITHO LEVEL 1', desc: 'Print circuit pattern with UV light' },
  { step: 'IMPLANT WELL', desc: 'Shoot dopant atoms into silicon' },
  { step: 'GATE OXIDE GROWTH', desc: 'Grow critical gate insulator layer' },
  { step: 'DEPOSIT POLYSILICON', desc: 'Add gate electrode material' },
  { step: 'CMP DIELECTRIC', desc: 'Polish surface flat for next layer' },
  { step: 'VIA ETCH', desc: 'Drill holes between metal layers' },
  { step: 'PARAMETRIC TEST', desc: 'Measure every device electrically' },
]

export default function LandingPage() {
  const navigate = useNavigate()
  const { setActiveBatch } = useAppStore()

  return (
    <div className="max-w-4xl mx-auto space-y-10 py-2">

      {/* What is this */}
      <div className="space-y-3">
        <div className="text-white/40 text-xs font-mono uppercase tracking-wider">AI Factory Austria · Infineon Track</div>
        <h1 className="text-3xl font-bold text-white leading-snug">
          Anomaly detection for<br />semiconductor process flows
        </h1>
        <p className="text-white/60 text-base leading-relaxed max-w-2xl">
          We trained three sequence models (GPT, BERT, LSTM) on Infineon manufacturing routes for IC, IGBT, and MOSFET chips.
          Given a process sequence, the models flag steps that are statistically unlikely to appear in that position — a signal that something went wrong.
        </p>
      </div>

      {/* Numbers */}
      <div className="grid grid-cols-4 gap-3">
        {[
          { n: '150k', label: 'training sequences' },
          { n: '208', label: 'unique process steps' },
          { n: '1,587', label: 'eval sequences scored' },
          { n: '99.1%', label: 'anomaly recall (BERT)' },
        ].map(({ n, label }) => (
          <div key={label} className="rounded-lg p-4 text-center border" style={{ backgroundColor: '#0d1426', borderColor: '#ffffff12' }}>
            <div className="text-2xl font-mono font-bold text-white">{n}</div>
            <div className="text-white/40 text-xs font-mono mt-1">{label}</div>
          </div>
        ))}
      </div>

      {/* Demo buttons */}
      <div className="space-y-3">
        <div className="text-white/40 text-xs font-mono uppercase tracking-wider">Run a demo</div>
        <div className="grid grid-cols-3 gap-4">
          <button
            onClick={() => { setActiveBatch(badBatch); navigate('/journey') }}
            className="rounded-xl border p-5 text-left transition-all hover:border-red-500/40 hover:bg-red-500/5"
            style={{ backgroundColor: '#0d1426', borderColor: '#ff444430' }}
          >
            <div className="font-mono font-bold text-white mb-1">Defective batch</div>
            <div className="text-white/40 text-xs font-mono">17 steps flagged · quality score 61.4%</div>
            <div className="text-white/30 text-xs font-mono mt-2">Shows the model catching anomalous steps in a bad IC process flow</div>
          </button>
          <button
            onClick={() => { setActiveBatch(goodBatch); navigate('/journey') }}
            className="rounded-xl border p-5 text-left transition-all hover:border-green-500/40 hover:bg-green-500/5"
            style={{ backgroundColor: '#0d1426', borderColor: '#00cc6630' }}
          >
            <div className="font-mono font-bold text-white mb-1">Normal batch</div>
            <div className="text-white/40 text-xs font-mono">2 steps flagged · quality score 94.2%</div>
            <div className="text-white/30 text-xs font-mono mt-2">A healthy process flow for comparison</div>
          </button>
          <button
            onClick={() => navigate('/models')}
            className="rounded-xl border p-5 text-left transition-all hover:border-purple-500/40 hover:bg-purple-500/5"
            style={{ backgroundColor: '#0d1426', borderColor: '#a78bfa30' }}
          >
            <div className="font-mono font-bold text-white mb-1">Model comparison</div>
            <div className="text-white/40 text-xs font-mono">GPT · BERT · LSTM</div>
            <div className="text-white/30 text-xs font-mono mt-2">Accuracy, recall, and size trade-offs across all three architectures</div>
          </button>
        </div>
      </div>

      {/* Real process steps */}
      <div className="space-y-3">
        <div className="text-white/40 text-xs font-mono uppercase tracking-wider">
          What a process sequence looks like — 10 of ~108 steps (IC route)
        </div>
        <div className="rounded-xl border overflow-hidden" style={{ backgroundColor: '#0d1426', borderColor: '#ffffff10' }}>
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left px-4 py-2 text-white/30 font-normal w-8">#</th>
                <th className="text-left px-4 py-2 text-white/30 font-normal">Step name</th>
                <th className="text-left px-4 py-2 text-white/30 font-normal">What it does</th>
              </tr>
            </thead>
            <tbody>
              {REAL_STEPS.map(({ step, desc }, i) => (
                <tr key={step} className="border-b border-white/5 last:border-0">
                  <td className="px-4 py-2 text-white/20">{i + 1}</td>
                  <td className="px-4 py-2 text-white/80">{step}</td>
                  <td className="px-4 py-2 text-white/40">{desc}</td>
                </tr>
              ))}
              <tr>
                <td className="px-4 py-2 text-white/20">…</td>
                <td colSpan={2} className="px-4 py-2 text-white/20">98 more steps</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="text-white/30 text-xs font-mono">
          The model learns the expected ordering and co-occurrence of these steps from 150k real sequences. At inference it flags any step whose position looks statistically unlikely.
        </div>
      </div>

    </div>
  )
}
