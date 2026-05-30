import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import KpiStrip from './components/KpiStrip'
import LandingPage from './pages/LandingPage'
import WaferJourneyPage from './pages/WaferJourneyPage'
import ModelComparisonPage from './pages/ModelComparisonPage'
import OptimizerPage from './pages/OptimizerPage'
import BatchInspectorPage from './pages/BatchInspectorPage'

const NAV = [
  { to: '/', label: 'Home', end: true },
  { to: '/journey', label: 'Wafer Inspector', end: false },
  { to: '/models', label: 'Model Race', end: false },
  { to: '/optimize', label: 'Optimizer', end: false },
  { to: '/batches', label: 'Batch Triage', end: false },
]

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-background text-white" style={{ backgroundColor: '#0a0f1e' }}>
        <KpiStrip />
        <nav className="flex gap-1 px-6 py-2 border-b border-white/10 bg-[#0a0f1e]">
          {NAV.map(({ to, label, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                `text-sm font-mono px-3 py-1.5 rounded transition-colors ${
                  isActive
                    ? 'bg-accent/20 text-accent'
                    : 'text-white/50 hover:text-white hover:bg-white/5'
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>
        <main className="p-6">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/journey" element={<WaferJourneyPage />} />
            <Route path="/models" element={<ModelComparisonPage />} />
            <Route path="/optimize" element={<OptimizerPage />} />
            <Route path="/batches" element={<BatchInspectorPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
