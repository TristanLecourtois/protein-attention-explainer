import { useState, useCallback, useRef, useEffect } from 'react'

const API = '/api'

export function useProteinData() {
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState(null)   // PENDING | RUNNING | DONE | ERROR
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState('')
  const [meta, setMeta] = useState(null)
  const [pdb, setPdb] = useState(null)
  const [residues, setResidues] = useState([])
  const [attentionData, setAttentionData] = useState(null)
  const [layerProfile, setLayerProfile] = useState(null)
  const [error, setError] = useState(null)

  const wsRef = useRef(null)
  const pollRef = useRef(null)

  // ------------------------------------------------------------------
  // Submit sequence
  // ------------------------------------------------------------------
  const submit = useCallback(async (sequence) => {
    setError(null)
    setPdb(null)
    setResidues([])
    setAttentionData(null)
    setLayerProfile(null)
    setProgress(0)
    setStatus('PENDING')
    setMessage('Submitting…')

    try {
      const resp = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence }),
      })

      if (!resp.ok) {
        const err = await resp.json()
        throw new Error(err.detail || `HTTP ${resp.status}`)
      }

      const job = await resp.json()
      setJobId(job.job_id)
      setStatus(job.status)
      setProgress(job.progress)
      setMessage(job.message)

      if (job.status === 'DONE') {
        await _loadResults(job.job_id)
      } else {
        _startPolling(job.job_id)
      }
    } catch (e) {
      setStatus('ERROR')
      setError(e.message)
    }
  }, [])

  // ------------------------------------------------------------------
  // Polling
  // ------------------------------------------------------------------
  const _startPolling = useCallback((id) => {
    if (pollRef.current) clearInterval(pollRef.current)

    pollRef.current = setInterval(async () => {
      try {
        const resp = await fetch(`${API}/jobs/${id}/status`)
        if (!resp.ok) return
        const job = await resp.json()
        setStatus(job.status)
        setProgress(job.progress)
        setMessage(job.message)

        if (job.status === 'DONE') {
          clearInterval(pollRef.current)
          await _loadResults(id)
        } else if (job.status === 'ERROR') {
          clearInterval(pollRef.current)
          setError(job.message)
        }
      } catch (e) {
        console.warn('Polling error:', e)
      }
    }, 1500)
  }, [])

  // ------------------------------------------------------------------
  // Load results
  // ------------------------------------------------------------------
  const _loadResults = useCallback(async (id) => {
    try {
      const [pdbResp, residuesResp, profileResp] = await Promise.all([
        fetch(`${API}/jobs/${id}/pdb`),
        fetch(`${API}/jobs/${id}/residues`),
        fetch(`${API}/jobs/${id}/layer-profile`),
      ])

      const pdbData = await pdbResp.json()
      const residuesData = await residuesResp.json()
      const profileData = await profileResp.json()

      setPdb(pdbData.pdb)
      setResidues(residuesData.residues)
      setLayerProfile(profileData)

      // Load attention for layer 0
      await loadAttention(id, 0)
    } catch (e) {
      setError(`Failed to load results: ${e.message}`)
    }
  }, [])

  // ------------------------------------------------------------------
  // Load attention for a specific layer
  // ------------------------------------------------------------------
  const loadAttention = useCallback(async (
    id,
    layer,
    threshold = 0.15,
    minSep = 6,
    maxEdges = 150,
  ) => {
    const targetId = id || jobId
    if (!targetId) return

    try {
      const resp = await fetch(
        `${API}/jobs/${targetId}/attention?layer=${layer}&threshold=${threshold}&min_separation=${minSep}&max_edges=${maxEdges}`
      )
      if (!resp.ok) return
      const data = await resp.json()
      setAttentionData(data)
    } catch (e) {
      console.warn('Attention fetch error:', e)
    }
  }, [jobId])

  // ------------------------------------------------------------------
  // Load PDB with attention B-factors
  // ------------------------------------------------------------------
  const loadPdbWithAttention = useCallback(async (layer) => {
    if (!jobId) return null
    try {
      const resp = await fetch(`${API}/jobs/${jobId}/pdb?attention_layer=${layer}`)
      if (!resp.ok) return null
      const data = await resp.json()
      return data.pdb
    } catch (e) {
      return null
    }
  }, [jobId])

  // Cleanup
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
      if (wsRef.current) wsRef.current.close()
    }
  }, [])

  return {
    jobId,
    status,
    progress,
    message,
    meta,
    pdb,
    residues,
    attentionData,
    layerProfile,
    error,
    submit,
    loadAttention,
    loadPdbWithAttention,
  }
}
