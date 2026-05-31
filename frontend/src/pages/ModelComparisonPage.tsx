import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, RadarChart, Radar, PolarGrid, PolarAngleAxis,
} from 'recharts'

// All numbers from:
//   procseq_base_d20000_s16001_seed11101/metrics.json
//   procseq_base_d20000_s16001_seed11101/metrics_hybrid.json

const LAYERS = [
  {
    id: 'decoder',
    name: 'Decoder (Tasks 1 & 2)',
    accent: '#00d4ff',
    params: '~580K',
    size: '~2 MB',
    stats: [
      { label: 'Task 1 Top-1', value: '87.7%' },
      { label: 'Task 1 Top-3', value: '100%' },
      { label: 'Task 1 Top-5', value: '100%' },
      { label: 'Task 1 MRR', value: '93.8%' },
      { label: 'Task 1 Category Top-1', value: '99.0%' },
      { label: 'Task 2 Block Accuracy', value: '93.7%' },
      { label: 'Task 2 Token Accuracy', value: '70.8%' },
      { label: 'Task 2 Rule Validity', value: '100%' },
    ],
  },
  {
    id: 'encoder',
    name: 'Encoder (Task 3, standalone)',
    accent: '#ffaa00',
    params: '~580K',
    size: '~2 MB',
    stats: [
      { label: 'Binary Accuracy', value: '60.8%' },
      { label: 'Precision', value: '0%' },
      { label: 'Recall', value: '0%' },
      { label: 'F1', value: '0.0' },
      { label: 'AUC', value: '48.7%' },
      { label: 'TP / FP / TN / FN', value: '0 / 0 / 600 / 387' },
    ],
  },
  {
    id: 'hybrid',
    name: 'Physics Hybrid (submitted)',
    accent: '#00cc66',
    params: '~580K + 10 rules',
    size: '~2 MB + ~50 KB',
    stats: [
      { label: 'Task 1 Top-1 (re-ranked)', value: '93.7%' },
      { label: 'Task 1 MRR', value: '96.8%' },
      { label: 'Task 1 Category Top-1', value: '99.8%' },
      { label: 'Task 3 Binary Accuracy', value: '100%' },
      { label: 'Task 3 Precision', value: '100%' },
      { label: 'Task 3 Recall', value: '100%' },
      { label: 'Task 3 F1', value: '1.0' },
      { label: 'Rule Attribution', value: '97.2%' },
    ],
  },
]

const RADAR_DATA = [
  { axis: 'Task1 Top-1',     Decoder: 88, Encoder: 61, Hybrid: 94 },
  { axis: 'Task2 Block Acc', Decoder: 94, Encoder: 0,  Hybrid: 94 },
  { axis: 'Rule Validity',   Decoder: 100, Encoder: 0, Hybrid: 100 },
  { axis: 'Task3 Accuracy',  Decoder: 61, Encoder: 61, Hybrid: 100 },
  { axis: 'Task3 F1',        Decoder: 0,  Encoder: 0,  Hybrid: 100 },
]

const TASK3_DATA = [
  { metric: 'Binary Acc', Encoder: 60.8, Hybrid: 100 },
  { metric: 'F1 Score', Encoder: 0, Hybrid: 100 },
  { metric: 'Rule Attr.', Encoder: 0, Hybrid: 97.2 },
]

export default function ModelComparisonPage() {
  return (
    <div className="max-w-6xl mx-auto space-y-8 py-2">

      <div>
        <h1 className="text-white font-bold text-2xl">Model Comparison</h1>
        <p className="text-white/40 text-xs font-mono mt-1">
          Three-task system: decoder (next-step & completion, Tasks 1–2) + physics engine (anomaly detection, Task 3). Trained on 20k sequences across IC/IGBT/MOSFET. 210+ token vocabulary.
        </p>
      </div>

      {/* Architecture diagram */}
      <div className="rounded-xl border p-4 font-mono text-xs space-y-4" style={{ backgroundColor: '#0d1426', borderColor: '#ffffff10' }}>
        <div className="text-white/40 uppercase tracking-wider">System architecture</div>

        {/* Tasks 1 & 2 path */}
        <div>
          <div className="text-white/25 text-xs mb-2">Tasks 1 & 2 — next-step prediction · sequence completion</div>
          <div className="flex items-center gap-2 overflow-x-auto pb-1">
            <div className="flex-shrink-0 rounded-lg px-3 py-2 border text-center min-w-[100px]" style={{ backgroundColor: '#060b18', borderColor: '#ffffff18' }}>
              <div className="text-white/50">Input</div>
              <div className="text-white/25 mt-0.5">partial sequence</div>
            </div>
            <div className="text-white/15 flex-shrink-0">→</div>
            <div className="flex-shrink-0 rounded-lg px-3 py-2 border text-center min-w-[160px]" style={{ backgroundColor: '#060b18', borderColor: '#00d4ff40' }}>
              <div style={{ color: '#00d4ff' }}>Decoder</div>
              <div className="text-white/30 mt-0.5">LlamaForCausalLM · 580K</div>
            </div>
            <div className="text-white/15 flex-shrink-0">→</div>
            <div className="flex-shrink-0 rounded-lg px-3 py-2 border text-center min-w-[150px]" style={{ backgroundColor: '#060b18', borderColor: '#00cc6640' }}>
              <div style={{ color: '#00cc66' }}>Physics engine</div>
              <div className="text-white/30 mt-0.5">re-rank / repair</div>
            </div>
            <div className="text-white/15 flex-shrink-0">→</div>
            <div className="flex-shrink-0 rounded-lg px-3 py-2 border text-center" style={{ backgroundColor: '#060b18', borderColor: '#a78bfa30' }}>
              <div style={{ color: '#a78bfa' }}>Top-5 / completion</div>
            </div>
          </div>
        </div>

        {/* Task 3 path */}
        <div>
          <div className="text-white/25 text-xs mb-2">Task 3 — anomaly detection</div>
          <div className="flex items-start gap-2 overflow-x-auto pb-1">
            <div className="flex-shrink-0 rounded-lg px-3 py-2 border text-center min-w-[100px]" style={{ backgroundColor: '#060b18', borderColor: '#ffffff18' }}>
              <div className="text-white/50">Input</div>
              <div className="text-white/25 mt-0.5">full sequence</div>
            </div>
            <div className="text-white/15 flex-shrink-0 mt-3">→</div>
            {/* Two branches */}
            <div className="flex flex-col gap-2 flex-shrink-0">
              {/* Encoder branch — failed */}
              <div className="flex items-center gap-2">
                <div className="rounded-lg px-3 py-2 border text-center min-w-[160px]" style={{ backgroundColor: '#060b18', borderColor: '#ffaa0030' }}>
                  <div style={{ color: '#ffaa00' }}>Encoder</div>
                  <div className="text-white/30 mt-0.5">DeBERTa-style · 580K</div>
                </div>
                <div className="text-white/15">→</div>
                <div className="rounded-lg px-3 py-1.5 border text-center" style={{ backgroundColor: '#ff444410', borderColor: '#ff444430' }}>
                  <div style={{ color: '#ff6666' }}>60.8% acc</div>
                  <div className="text-white/25 mt-0.5">collapses to all-valid</div>
                </div>
                <div
                  className="text-xs px-2 py-0.5 rounded flex-shrink-0"
                  style={{ backgroundColor: '#ff444415', color: '#ff666660', border: '1px solid #ff444425' }}
                >
                  ✗ not submitted
                </div>
              </div>
              {/* Physics branch — submitted */}
              <div className="flex items-center gap-2">
                <div className="rounded-lg px-3 py-2 border text-center min-w-[160px]" style={{ backgroundColor: '#060b18', borderColor: '#00cc6640' }}>
                  <div style={{ color: '#00cc66' }}>Physics engine</div>
                  <div className="text-white/30 mt-0.5">10 deterministic rules</div>
                </div>
                <div className="text-white/15">→</div>
                <div className="rounded-lg px-3 py-1.5 border text-center" style={{ backgroundColor: '#00cc6610', borderColor: '#00cc6640' }}>
                  <div style={{ color: '#00cc66' }}>100% acc</div>
                  <div className="text-white/25 mt-0.5">exact rule attribution</div>
                </div>
                <div
                  className="text-xs px-2 py-0.5 rounded flex-shrink-0"
                  style={{ backgroundColor: '#00cc6615', color: '#00cc66', border: '1px solid #00cc6630' }}
                >
                  ✓ submitted
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Layer cards */}
      <div className="grid lg:grid-cols-3 md:grid-cols-2 grid-cols-1 gap-5">
        {LAYERS.map((m) => (
          <div
            key={m.id}
            className="rounded-xl border p-4 space-y-3"
            style={{ backgroundColor: '#0d1426', borderColor: `${m.accent}28` }}
          >
            <div>
              <div className="font-mono font-bold text-white text-sm">{m.name}</div>
              <div className="text-white/30 text-xs font-mono mt-0.5">{m.params} params · {m.size}</div>
            </div>

            <div className="space-y-0">
              {m.stats.map((s) => (
                <div key={s.label} className="flex justify-between text-xs font-mono py-1.5 border-b border-white/5 last:border-0">
                  <span className="text-white/40">{s.label}</span>
                  <span className="text-white font-bold ml-2 text-right">{s.value}</span>
                </div>
              ))}
            </div>

          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-5">
        <div className="rounded-xl border p-4 space-y-2" style={{ backgroundColor: '#0d1426', borderColor: '#ffffff10' }}>
          <div className="font-mono text-sm text-white">Task 3: Encoder alone vs. Physics Hybrid</div>
          <div className="text-white/30 text-xs font-mono">Encoder collapses to all-valid prediction; hybrid achieves 100%.</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={TASK3_DATA} margin={{ top: 4, right: 10, left: -10, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
              <XAxis dataKey="metric" tick={{ fill: '#ffffff60', fontSize: 10, fontFamily: 'monospace' }} />
              <YAxis tick={{ fill: '#ffffff60', fontSize: 10, fontFamily: 'monospace' }} domain={[0, 100]} />
              <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #ffffff20', fontFamily: 'monospace', fontSize: 11 }} />
              <Legend wrapperStyle={{ fontSize: 10, fontFamily: 'monospace', color: '#ffffff60' }} />
              <Bar dataKey="Encoder" fill="#ffaa00" radius={[3, 3, 0, 0]} />
              <Bar dataKey="Hybrid" fill="#00cc66" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="rounded-xl border p-4 space-y-2" style={{ backgroundColor: '#0d1426', borderColor: '#ffffff10' }}>
          <div className="font-mono text-sm text-white">Capability radar (normalised)</div>
          <div className="text-white/30 text-xs font-mono">Physics has no next-step accuracy — it only validates, not predicts.</div>
          <ResponsiveContainer width="100%" height={200}>
            <RadarChart data={RADAR_DATA} margin={{ top: 10, right: 30, left: 30, bottom: 10 }}>
              <PolarGrid stroke="#ffffff12" />
              <PolarAngleAxis dataKey="axis" tick={{ fill: '#ffffff50', fontSize: 9, fontFamily: 'monospace' }} />
              <Radar name="Decoder" dataKey="Decoder" stroke="#00d4ff" fill="#00d4ff" fillOpacity={0.12} />
              <Radar name="Encoder" dataKey="Encoder" stroke="#ffaa00" fill="#ffaa00" fillOpacity={0.12} />
              <Radar name="Hybrid" dataKey="Hybrid" stroke="#00cc66" fill="#00cc66" fillOpacity={0.12} />
              <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'monospace', color: '#ffffff60' }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Summary table */}
      <div className="rounded-xl border overflow-hidden" style={{ backgroundColor: '#0d1426', borderColor: '#ffffff10' }}>
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="border-b border-white/10 bg-[#060b18]">
              {['Model', 'Params', 'Task 1 Top-1', 'Task 2 Block Acc', 'Task 3 Acc', 'Submitted'].map((h) => (
                <th key={h} className="text-left px-4 py-3 text-white/40 font-normal">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-white/5">
              <td className="px-4 py-3 text-white font-bold">Decoder</td>
              <td className="px-4 py-3 text-white/70">~580K</td>
              <td className="px-4 py-3 text-white/70">87.7%</td>
              <td className="px-4 py-3 text-white/70">93.7%</td>
              <td className="px-4 py-3" style={{ color: '#ffaa00' }}>60.8% (encoder)</td>
              <td className="px-4 py-3 text-white/40">—</td>
            </tr>
            <tr>
              <td className="px-4 py-3 text-white font-bold">Hybrid</td>
              <td className="px-4 py-3 text-white/70">~580K + rules</td>
              <td className="px-4 py-3 text-white/70">93.7%</td>
              <td className="px-4 py-3 text-white/70">93.7%</td>
              <td className="px-4 py-3" style={{ color: '#00cc66' }}>100%</td>
              <td className="px-4 py-3" style={{ color: '#00cc66' }}>✓</td>
            </tr>
          </tbody>
        </table>
      </div>

    </div>
  )
}
