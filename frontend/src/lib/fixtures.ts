import badBatchRaw from '../fixtures/bad-batch.json'
import goodBatchRaw from '../fixtures/good-batch.json'
import type { PredictResponse } from '../types/api'

export const badBatch: PredictResponse = badBatchRaw as unknown as PredictResponse
export const goodBatch: PredictResponse = goodBatchRaw as unknown as PredictResponse
