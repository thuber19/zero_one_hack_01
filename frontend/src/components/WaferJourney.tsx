import { useRef, useEffect, useState } from 'react'
import * as d3 from 'd3'
import type { Step } from '../types/api'
import { useAppStore } from '../lib/store'
import StepPopover from './StepPopover'

interface Props {
  steps: Step[]
}

const NODE_SPACING = 18
const NODE_RADIUS = 6
const SVG_HEIGHT = 80
const MARGIN = { top: 15, left: 20, right: 20, bottom: 15 }

function riskColor(riskScore: number): string {
  if (riskScore >= 0.85) return '#ff4444'
  if (riskScore >= 0.70) return '#ffaa00'
  return '#1e2a3a'
}

function riskGlow(riskScore: number): string {
  if (riskScore >= 0.85) return 'drop-shadow(0 0 6px #ff4444)'
  if (riskScore >= 0.70) return 'drop-shadow(0 0 4px #ffaa00)'
  return 'none'
}

export default function WaferJourney({ steps }: Props) {
  const svgRef = useRef<SVGSVGElement>(null)
  const { setSelectedStep } = useAppStore()
  const [popover, setPopover] = useState<{ step: Step; x: number; y: number } | null>(null)

  const svgWidth = steps.length * NODE_SPACING + MARGIN.left + MARGIN.right

  useEffect(() => {
    if (!svgRef.current || !steps.length) return
    const svg = d3.select(svgRef.current)
    const y = SVG_HEIGHT / 2

    svg.selectAll<SVGCircleElement, Step>('circle.step-node')
      .data(steps, (d) => d.step_id)
      .join(
        (enter) =>
          enter
            .append('circle')
            .attr('class', 'step-node')
            .attr('r', NODE_RADIUS)
            .attr('cx', (_, i) => MARGIN.left + i * NODE_SPACING)
            .attr('cy', y)
            .attr('fill', (d) => riskColor(d.risk_score))
            .attr('stroke', (d) => (d.risk_score >= 0.70 ? riskColor(d.risk_score) : '#2a3a4a'))
            .attr('stroke-width', 1)
            .style('filter', (d) => riskGlow(d.risk_score))
            .style('cursor', 'pointer'),
        (update) =>
          update
            .attr('cx', (_, i) => MARGIN.left + i * NODE_SPACING)
            .attr('fill', (d) => riskColor(d.risk_score))
            .style('filter', (d) => riskGlow(d.risk_score)),
        (exit) => exit.remove()
      )
      .on('mouseover', function (event, d) {
        const rect = (event.target as SVGCircleElement).getBoundingClientRect()
        setPopover({ step: d, x: rect.left + rect.width / 2, y: rect.top })
        setSelectedStep(d)
      })
      .on('mouseleave', function () {
        setPopover(null)
      })
      .on('click', function (_, d) {
        setSelectedStep(d)
      })
  }, [steps, setSelectedStep])

  return (
    <div>
      <div style={{ overflowX: 'auto', overflowY: 'visible' }}>
        <svg
          ref={svgRef}
          width={svgWidth}
          height={SVG_HEIGHT}
          style={{ display: 'block' }}
        />
      </div>
      {popover && (
        <StepPopover
          step={popover.step}
          x={popover.x}
          y={popover.y}
          onClose={() => setPopover(null)}
        />
      )}
    </div>
  )
}
