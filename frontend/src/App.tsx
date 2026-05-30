import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import KpiStrip from './components/KpiStrip'
import WaferJourneyPage from './pages/WaferJourneyPage'
import ShapPage from './pages/ShapPage'
import OptimizerPage from './pages/OptimizerPage'
import BatchInspectorPage from './pages/BatchInspectorPage'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-background text-white" style={{ backgroundColor: '#0a0f1e' }}>
        <KpiStrip />
        <nav className="flex gap-4 px-6 py-2 border-b border-white/10 bg-[#0a0f1e]">
          {[
            { to: '/', label: 'Wafer Journey' },
            { to: '/shap', label: 'SHAP Panel' },
            { to: '/optimize', label: 'Optimizer' },
            { to: '/batches', label: 'Batch Inspector' },
          ].map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `text-sm font-mono px-3 py-1 rounded transition-colors ${
                  isActive ? 'bg-accent/20 text-accent' : 'text-white/60 hover:text-white'
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>
        <main className="p-6">
          <Routes>
            <Route path="/" element={<WaferJourneyPage />} />
            <Route path="/shap" element={<ShapPage />} />
            <Route path="/optimize" element={<OptimizerPage />} />
            <Route path="/batches" element={<BatchInspectorPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
