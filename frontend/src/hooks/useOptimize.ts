import { useMutation } from '@tanstack/react-query'
import { useRef, useCallback } from 'react'
import { postOptimize } from '../lib/api'
import { useToast } from '../components/ui/toast'
import type { OptimizeResponse } from '../types/api'

interface OptimizeStep {
  step_id: string
  step_name: string
  category: string
}

export function useOptimize() {
  const { addToast } = useToast()
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const mutation = useMutation<OptimizeResponse, Error, OptimizeStep[]>({
    mutationFn: (sequence) => postOptimize({ sequence }),
    onError: (err) => {
      const message = err.name === 'AbortError' || err.message.includes('abort')
        ? 'Optimize request timed out — using last result'
        : 'Optimize failed — try again'
      addToast(message, 'warning')
    },
  })

  const optimize = useCallback((sequence: OptimizeStep[]) => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      mutation.mutate(sequence)
    }, 200)
  }, [mutation])

  return {
    optimize,
    predictedYield: mutation.data?.predicted_yield ?? null,
    isLoading: mutation.isPending,
    isError: mutation.isError,
  }
}
