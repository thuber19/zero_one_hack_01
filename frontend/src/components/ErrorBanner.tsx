import { useAppStore } from '../lib/store'

export default function ErrorBanner() {
  const isDegradedMode = useAppStore((s) => s.isDegradedMode)
  if (!isDegradedMode) return null
  return (
    <div className="bg-risk-amber/20 border border-risk-amber text-risk-amber px-4 py-2 text-sm font-mono text-center">
      ⚠ Model API unavailable — displaying cached data
    </div>
  )
}
