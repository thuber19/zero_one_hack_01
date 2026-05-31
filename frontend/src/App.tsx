import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import WaferJourneyPage from './pages/WaferJourneyPage'
import ModelComparisonPage from './pages/ModelComparisonPage'
import OptimizerPage from './pages/OptimizerPage'
import BatchInspectorPage from './pages/BatchInspectorPage'
import InferencePage from './pages/InferencePage'

const NAV = [
  { to: '/', label: 'Home', end: true },
  { to: '/journey', label: 'Sequence Analysis', end: false },
  { to: '/models', label: 'Model Race', end: false },
  { to: '/batches', label: 'Eval Browser', end: false },
]

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-background text-white" style={{ backgroundColor: '#0a0f1e' }}>
        <div className="flex flex-col gap-2 px-6 py-2 border-b border-white/10 bg-[#0a0f1e] sm:flex-row sm:items-center sm:justify-between">
          <span className="text-accent font-mono text-sm font-bold tracking-widest uppercase">
            Team TBD: ProcSeq Monitor
          </span>
          <nav className="flex flex-wrap gap-1">
            {NAV.map(({ to, label, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) =>
                  `text-sm font-mono px-3 py-1.5 rounded transition-colors border-b-2 ${
                    isActive
                      ? 'bg-accent/20 text-accent border-accent'
                      : 'text-white/50 hover:text-white hover:bg-white/5 border-transparent'
                  }`
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>
        </div>
        <main className="p-6">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/journey" element={<WaferJourneyPage />} />
            <Route path="/models" element={<ModelComparisonPage />} />
            <Route path="/optimize" element={<OptimizerPage />} />
            <Route path="/batches" element={<BatchInspectorPage />} />
            <Route path="/infer" element={<InferencePage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
