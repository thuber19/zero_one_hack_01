import { useQuery } from '@tanstack/react-query'
import { getBatchDetail } from '../lib/api'
import type { PredictResponse } from '../types/api'

export function useBatchDetail(batchId: string | null) {
  return useQuery<PredictResponse>({
    queryKey: ['batch-detail', batchId],
    queryFn: () => getBatchDetail(batchId!),
    enabled: batchId !== null,
    staleTime: Infinity,
    retry: 1,
  })
}
