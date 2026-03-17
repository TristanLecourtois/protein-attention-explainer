/**
 * ProteinViewer — Mol* (Molstar) 3D viewer + 2D attention edge overlay
 *
 * Mol* handles the protein cartoon (colored by B-factor = attention score).
 * A transparent <canvas> overlay draws attention edges projected through
 * Mol*'s live camera matrices — perfect sync with 3D rotation/zoom.
 */

import { useEffect, useRef, useCallback, useState } from 'react'

export default function ProteinViewer({
  pdb,
  residues,
  attentionData,
  selectedResidue,
  onResidueClick,
  colorMode = 'attention',
  showEdges = true,
  isLoading = false,
}) {
  const containerRef = useRef(null)
  const canvasRef    = useRef(null)   // Mol* WebGL canvas
  const overlayRef   = useRef(null)   // 2D edge overlay canvas
  const pluginRef    = useRef(null)
  const rafRef       = useRef(null)

  // Keep latest values available inside rAF without re-registering the loop
  const residuesRef  = useRef(residues)
  const edgesRef     = useRef([])
  const showEdgesRef = useRef(showEdges)
  const [ready, setReady] = useState(false)
  const [viewerError, setViewerError] = useState(null)

  useEffect(() => { residuesRef.current = residues }, [residues])
  useEffect(() => {
    edgesRef.current   = attentionData?.edges || []
    showEdgesRef.current = showEdges
  }, [attentionData, showEdges])


  const drawOverlay = useCallback(() => {
    const overlay = overlayRef.current
    const plugin  = pluginRef.current
    if (!overlay || !plugin?.canvas3d) return

    const w = overlay.clientWidth  * window.devicePixelRatio
    const h = overlay.clientHeight * window.devicePixelRatio
    if (overlay.width !== w)  overlay.width  = w
    if (overlay.height !== h) overlay.height = h

    const ctx = overlay.getContext('2d')
    ctx.clearRect(0, 0, w, h)

    if (!showEdgesRef.current) return
    const edges = edgesRef.current
    const res   = residuesRef.current
    if (!edges.length || !res.length) return

    // camera.projectionView is a column-major Float32Array(16) (proj × view)
    const pv = plugin.canvas3d.camera.projectionView

    const project = (x, y, z) => {
      const cx = pv[0]*x + pv[4]*y + pv[8]*z  + pv[12]
      const cy = pv[1]*x + pv[5]*y + pv[9]*z  + pv[13]
      const cw = pv[3]*x + pv[7]*y + pv[11]*z + pv[15]
      if (cw <= 0) return null
      return {
        x: ( cx/cw * 0.5 + 0.5) * w,
        y: (1 - (cy/cw * 0.5 + 0.5)) * h,
      }
    }

    edges.forEach(({ i, j, weight }) => {
      const ri = res[i], rj = res[j]
      if (!ri || !rj) return
      const p1 = project(ri.x, ri.y, ri.z)
      const p2 = project(rj.x, rj.y, rj.z)
      if (!p1 || !p2) return

      const r = Math.round(26  + weight * 229)
      const g = Math.round(100 - weight * 40)
      const b = Math.round(220 - weight * 180)

      ctx.beginPath()
      ctx.moveTo(p1.x, p1.y)
      ctx.lineTo(p2.x, p2.y)
      ctx.strokeStyle = `rgba(${r},${g},${b},${0.35 + weight * 0.55})`
      ctx.lineWidth   = (0.6 + weight * 2.4) * window.devicePixelRatio
      ctx.lineCap     = 'round'
      ctx.stroke()
    })
  }, [])


  useEffect(() => {
    if (!containerRef.current || !canvasRef.current) return
    let plugin = null

    ;(async () => {
      try {
        const [
          { PluginContext },
          { DefaultPluginSpec },
          { Color },
        ] = await Promise.all([
          import('molstar/lib/mol-plugin/context'),
          import('molstar/lib/mol-plugin/spec'),
          import('molstar/lib/mol-util/color'),
        ])

        const spec = DefaultPluginSpec()

        // Override renderer background + hide all Mol* UI panels
        spec.canvas3d = {
          ...spec.canvas3d,
          renderer: {
            ...spec.canvas3d?.renderer,
            backgroundColor: Color(0x050810),
          },
          camera: {
            helper: { axes: { name: 'off', params: {} } },
          },
        }
        spec.layout = {
          initial: {
            isExpanded: false,
            showControls: false,
            regionState: {
              bottom: 'hidden',
              left:   'hidden',
              right:  'hidden',
              top:    'hidden',
            },
          },
        }

        plugin = new PluginContext(spec)
        await plugin.init()
        plugin.initViewer(canvasRef.current, containerRef.current)

        pluginRef.current = plugin
        setReady(true)

        // Click → residue index
        plugin.behaviors.interaction.click.subscribe(({ current }) => {
          if (!current?.loci) return
          _handleClick(current.loci, residuesRef.current, onResidueClick)
        })

        // rAF loop for edge overlay (also redraws on camera move)
        const loop = () => { drawOverlay(); rafRef.current = requestAnimationFrame(loop) }
        rafRef.current = requestAnimationFrame(loop)

      } catch (err) {
        console.error('[Molstar] init failed:', err)
        setViewerError(String(err))
      }
    })()

    // Resize observer
    const ro = new ResizeObserver(() => pluginRef.current?.handleResize?.())
    ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      if (plugin) plugin.dispose()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps


  useEffect(() => {
    if (!ready || !pdb || !pluginRef.current) return
    loadStructure(pluginRef.current, pdb, colorMode)
  }, [ready, pdb]) // eslint-disable-line react-hooks/exhaustive-deps


  useEffect(() => {
    if (!ready || !pluginRef.current || !pdb) return
    applyColorTheme(pluginRef.current, colorMode)
  }, [ready, colorMode]) // eslint-disable-line react-hooks/exhaustive-deps


  useEffect(() => {
    if (!ready || !pluginRef.current) return
    if (selectedResidue == null) return
    const r = residuesRef.current[selectedResidue]
    if (r) focusResidue(pluginRef.current, r.resno, r.chain)
  }, [ready, selectedResidue])

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', height: '100%', position: 'relative', overflow: 'hidden' }}
    >
      {/* Mol* WebGL canvas */}
      <canvas
        ref={canvasRef}
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
      />

      {/* Attention edge overlay */}
      <canvas
        ref={overlayRef}
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
      />

      {isLoading && (
        <div className="viewer-overlay">
          <div className="loading-spinner" />
          <span className="viewer-overlay-text">Running ESMFold…</span>
        </div>
      )}

      {viewerError && (
        <div className="viewer-overlay">
          <div style={{ color: 'var(--accent-pink)', fontFamily: 'var(--font-mono)', fontSize: 12, maxWidth: 400, padding: 20, textAlign: 'center' }}>
            ⚠ Mol* error: {viewerError}
          </div>
        </div>
      )}

      {!pdb && !isLoading && !viewerError && (
        <div className="viewer-overlay">
          <div className="viewer-welcome">
            <div className="dna-icon">🧬</div>
            <h2>Protein Attention Explainer</h2>
            <p>
              Enter an amino acid sequence on the left to predict its 3D structure
              and visualize ESMFold attention maps with <strong>Mol*</strong>.
            </p>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: 8 }}>
              ESM-2 · 33 layers · 20 heads
            </div>
          </div>
        </div>
      )}
    </div>
  )
}


async function loadStructure(plugin, pdb, colorMode) {
  try {
    await plugin.clear()

    const data = await plugin.builders.data.rawData(
      { data: pdb, label: 'protein' },
      { state: { isGhost: true } }
    )

    const traj = await plugin.builders.structure.parseTrajectory(data, 'pdb')

    // Use 'default' preset — the simplest and most reliable
    await plugin.builders.structure.hierarchy.applyPreset(traj, 'default')

    // Small delay to let representations settle, then apply color theme
    await new Promise(r => setTimeout(r, 100))
    await applyColorTheme(plugin, colorMode)

  } catch (err) {
    console.error('[Molstar] loadStructure failed:', err)
  }
}

async function applyColorTheme(plugin, colorMode) {
  try {
    const structures = plugin.managers.structure.hierarchy.current.structures
    if (!structures.length) return

    const theme =
      colorMode === 'attention' ? 'uncertainty'
      : colorMode === 'bfactor' ? 'uncertainty'
      : colorMode === 'chainid' ? 'chain-id'
      : 'element-symbol'

    for (const s of structures) {
      await plugin.managers.structure.component.updateRepresentationsTheme(
        s.components,
        { color: theme }
      )
    }
  } catch (err) {
    console.error('[Molstar] applyColorTheme failed:', err)
  }
}

async function focusResidue(plugin, resno, chain) {
  try {
    const { MolScriptBuilder: MS } = await import('molstar/lib/mol-script/language/builder')
    const { compile }              = await import('molstar/lib/mol-script/runtime/query/compiler')
    const { StructureSelection }   = await import('molstar/lib/mol-model/structure')

    const structures = plugin.managers.structure.hierarchy.current.structures
    if (!structures.length) return
    const structure = structures[0].cell.obj?.data
    if (!structure) return

    const expr = MS.struct.generator.atomGroups({
      'residue-test': MS.core.rel.eq([
        MS.struct.atomProperty.macromolecular.auth_seq_id(), resno,
      ]),
      'chain-test': MS.core.rel.eq([
        MS.struct.atomProperty.macromolecular.auth_asym_id(), chain,
      ]),
    })

    const { QueryContext } = await import('molstar/lib/mol-model/structure')
    const result = compile(expr)(new QueryContext(structure))
    const loci   = StructureSelection.toLociWithSourceUnits(result)

    plugin.managers.structure.focus.setFromLoci(loci)
    plugin.managers.camera.focusLoci(loci, { durationMs: 400 })
  } catch (err) {
    console.error('[Molstar] focusResidue failed:', err)
  }
}

async function _handleClick(loci, residues, onResidueClick) {
  if (!onResidueClick) return
  try {
    const { StructureElement } = await import('molstar/lib/mol-model/structure')
    if (!StructureElement.Loci.is(loci)) return

    const loc = StructureElement.Location.create(loci.structure)
    for (const e of loci.elements) {
      StructureElement.Location.set(loc, loci.structure, e.unit, e.indices[0])
      const { StructureProperties: P } = await import('molstar/lib/mol-model/structure')
      const seqId = P.residue.auth_seq_id(loc)
      const chain = P.chain.auth_asym_id(loc)
      const idx   = residues.findIndex(r => r.resno === seqId && r.chain === chain)
      if (idx >= 0) onResidueClick(idx)
      break
    }
  } catch (_) {
    // soft fail
  }
}
