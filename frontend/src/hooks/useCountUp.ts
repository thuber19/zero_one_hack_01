import { useEffect, useRef, useState } from 'react'

function easeOutQuart(t: number): number {
  return 1 - Math.pow(1 - t, 4)
}

export function useCountUp(target: number, duration = 800): number {
  const [displayed, setDisplayed] = useState(target)
  const prevRef = useRef(target)
  const rafRef = useRef<number | null>(null)
  const startTimeRef = useRef<number | null>(null)
  const startValueRef = useRef(target)

  useEffect(() => {
    if (prevRef.current === target) return

    const from = prevRef.current
    prevRef.current = target
    startValueRef.current = from
    startTimeRef.current = null

    if (rafRef.current != null) cancelAnimationFrame(rafRef.current)

    const animate = (now: number) => {
      if (startTimeRef.current == null) startTimeRef.current = now
      const elapsed = now - startTimeRef.current
      const progress = Math.min(elapsed / duration, 1)
      const eased = easeOutQuart(progress)
      setDisplayed(from + (target - from) * eased)
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(animate)
      } else {
        setDisplayed(target)
      }
    }

    rafRef.current = requestAnimationFrame(animate)

    const handleVisibility = () => {
      if (document.hidden === false) {
        setDisplayed(target)
        if (rafRef.current != null) cancelAnimationFrame(rafRef.current)
      }
    }
    document.addEventListener('visibilitychange', handleVisibility)

    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current)
      document.removeEventListener('visibilitychange', handleVisibility)
    }
  }, [target, duration])

  return displayed
}
