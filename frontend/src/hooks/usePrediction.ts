import { useMutation } from '@tanstack/react-query'
import { postPredict } from '../lib/api'
import { badBatch } from '../lib/fixtures'
import { useAppStore } from '../lib/store'
import type { PredictResponse } from '../types/api'

export function usePrediction() {
  const { setActiveBatch, setIsDegradedMode } = useAppStore()

  return useMutation<PredictResponse, Error, { batchId?: string; sequence?: unknown[] }>({
    mutationFn: async ({ batchId, sequence }) => {
      return postPredict({ batch_id: batchId, sequence })
    },
    onSuccess: (data) => {
      setActiveBatch(data)
      setIsDegradedMode(false)
    },
    onError: () => {
      setActiveBatch(badBatch)
      setIsDegradedMode(true)
    },
  })
}
