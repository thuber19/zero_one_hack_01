import { useRef, useEffect, useState } from 'react'
import * as d3 from 'd3'
import type { Step } from '../types/api'
import { useAppStore } from '../lib/store'
import StepPopover from './StepPopover'

interface Props {
  steps: Step[]
}

const NODE_SPACING = 26
const NODE_RADIUS = 10
const SVG_HEIGHT = 130
const MARGIN = { top: 36, left: 20, right: 20, bottom: 28 }

const CATEGORY_STROKE: Record<string, string> = {
  cleaning: '#00d4ff',
  deposition: '#a78bfa',
  lithography: '#f59e0b',
  etch: '#34d399',
  anneal: '#f97316',
  implant: '#60a5fa',
  inspection: '#e879f9',
  planarization: '#94a3b8',
  test: '#e879f9',
  process: '#334155',
}

function catStroke(cat: string): string {
  return CATEGORY_STROKE[cat.toLowerCase()] ?? '#334155'
}

function riskFill(r: number): string {
  if (r >= 0.85) return '#ff4444'
  if (r >= 0.70) return '#ffaa00'
  return '#1e2a3a'
}

export default function WaferJourney({ steps }: Props) {
  const svgRef = useRef<SVGSVGElement>(null)
  const { setSelectedStep } = useAppStore()
  const [popover, setPopover] = useState<{ step: Step; x: number; y: number } | null>(null)

  const svgWidth = Math.max(400, steps.length * NODE_SPACING + MARGIN.left + MARGIN.right)

  useEffect(() => {
    if (!svgRef.current || !steps.length) return
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const cy = SVG_HEIGHT / 2

    // Glow filter def
    const defs = svg.append('defs')
    const fRed = defs.append('filter').attr('id', 'glow-red').attr('x', '-100%').attr('y', '-100%').attr('width', '300%').attr('height', '300%')
    fRed.append('feGaussianBlur').attr('stdDeviation', '4').attr('result', 'blur')
    const mRed = fRed.append('feMerge')
    mRed.append('feMergeNode').attr('in', 'blur')
    mRed.append('feMergeNode').attr('in', 'SourceGraphic')

    const fAmber = defs.append('filter').attr('id', 'glow-amber').attr('x', '-100%').attr('y', '-100%').attr('width', '300%').attr('height', '300%')
    fAmber.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'blur')
    const mAmber = fAmber.append('feMerge')
    mAmber.append('feMergeNode').attr('in', 'blur')
    mAmber.append('feMergeNode').attr('in', 'SourceGraphic')

    // Defect zone highlight
    const anomalousIndices = steps
      .map((s, i) => ({ s, i }))
      .filter(({ s }) => s.risk_score >= 0.70)
      .map(({ i }) => i)

    if (anomalousIndices.length > 0) {
      const minI = anomalousIndices[0]
      const maxI = anomalousIndices[anomalousIndices.length - 1]
      const x1 = MARGIN.left + minI * NODE_SPACING - NODE_RADIUS - 8
      const x2 = MARGIN.left + maxI * NODE_SPACING + NODE_RADIUS + 8

      svg.append('rect')
        .attr('x', x1)
        .attr('y', 6)
        .attr('width', Math.max(x2 - x1, 20))
        .attr('height', SVG_HEIGHT - 12)
        .attr('rx', 8)
        .attr('fill', '#ff444412')
        .attr('stroke', '#ff444428')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4,3')

      svg.append('text')
        .attr('x', (x1 + x2) / 2)
        .attr('y', 18)
        .attr('text-anchor', 'middle')
        .attr('font-size', 9)
        .attr('font-family', 'monospace')
        .attr('fill', '#ff4444aa')
        .text(`⚠ ${anomalousIndices.length} Issues Detected`)
    }

    // Connector line
    svg.append('line')
      .attr('x1', MARGIN.left)
      .attr('y1', cy)
      .attr('x2', MARGIN.left + (steps.length - 1) * NODE_SPACING)
      .attr('y2', cy)
      .attr('stroke', '#1e2a3a')
      .attr('stroke-width', 2)

    // Step index milestones
    steps.forEach((_, i) => {
      if (i % 10 === 0) {
        const x = MARGIN.left + i * NODE_SPACING
        svg.append('line')
          .attr('x1', x).attr('y1', cy + NODE_RADIUS + 6)
          .attr('x2', x).attr('y2', cy + NODE_RADIUS + 10)
          .attr('stroke', '#ffffff20').attr('stroke-width', 1)
        svg.append('text')
          .attr('x', x)
          .attr('y', SVG_HEIGHT - 4)
          .attr('text-anchor', 'middle')
          .attr('font-size', 8)
          .attr('font-family', 'monospace')
          .attr('fill', '#ffffff30')
          .text(i === 0 ? 'Start' : `Step ${i}`)
      }
    })

    // Nodes
    svg.selectAll<SVGGElement, Step>('g.node')
      .data(steps, (d) => d.step_id)
      .join((enter) => {
        const g = enter.append('g')
          .attr('class', 'node')
          .attr('transform', (_, i) => `translate(${MARGIN.left + i * NODE_SPACING},${cy})`)
          .style('cursor', 'pointer')
          .style('opacity', 0)

        // Category ring
        g.append('circle')
          .attr('r', NODE_RADIUS + 4)
          .attr('fill', 'none')
          .attr('stroke', (d) => catStroke(d.category))
          .attr('stroke-width', 1.5)
          .attr('opacity', (d) => d.risk_score >= 0.70 ? 0.8 : 0.35)

        // Main fill
        g.append('circle')
          .attr('r', NODE_RADIUS)
          .attr('fill', (d) => riskFill(d.risk_score))
          .attr('stroke', (d) => d.risk_score >= 0.70 ? riskFill(d.risk_score) : '#2a3a4a')
          .attr('stroke-width', 1.5)

        // Glow filter on risky nodes
        g.filter((d) => d.risk_score >= 0.85).attr('filter', 'url(#glow-red)')
        g.filter((d) => d.risk_score >= 0.70 && d.risk_score < 0.85).attr('filter', 'url(#glow-amber)')

        // Staggered entry animation
        g.transition()
          .delay((_, i) => Math.min(i * 4, 400))
          .duration(250)
          .style('opacity', 1)

        return g
      },
      (update) => {
        update.attr('transform', (_, i) => `translate(${MARGIN.left + i * NODE_SPACING},${cy})`)
        update.select('circle:nth-child(2)')
          .attr('fill', (d) => riskFill(d.risk_score))
        return update
      },
      (exit) => exit.transition().duration(150).style('opacity', 0).remove()
      )
      .on('mouseover', function (event, d) {
        const rect = (event.currentTarget as SVGGElement).getBoundingClientRect()
        setPopover({ step: d, x: rect.left + rect.width / 2, y: rect.top })
        setSelectedStep(d)
      })
      .on('mouseleave', () => setPopover(null))
      .on('click', (_, d) => setSelectedStep(d))
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
