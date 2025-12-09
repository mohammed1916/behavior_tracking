import React, { useEffect, useState, useRef } from 'react';

export default function AnalysisDetails({ analysisId, onClose }) {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);
  const pauseTimerRef = useRef(null);

  function playRange(startSec, endSec) {
    const v = videoRef.current;
    if (!v) return;
    if (pauseTimerRef.current) { clearTimeout(pauseTimerRef.current); pauseTimerRef.current = null; }
    v.currentTime = startSec || 0;
    const dur = Math.max(0.2, (endSec || (startSec + 1)) - startSec);
    v.play().catch(() => {});
    pauseTimerRef.current = setTimeout(() => { try { v.pause(); } catch (e) {} pauseTimerRef.current = null; }, Math.ceil(dur * 1000) + 150);
  }

  useEffect(() => {
    if (!analysisId) return;
    setLoading(true);
    setError(null);
    fetch(`http://localhost:8001/backend/analysis/${encodeURIComponent(analysisId)}`)
      .then(async (r) => {
        if (!r.ok) throw new Error(await r.text());
        return r.json();
      })
      .then((data) => setAnalysis(data))
      .catch((e) => setError(e.message || String(e)))
      .finally(() => setLoading(false));
  }, [analysisId]);

  if (!analysisId) return null;

  return (
    <div className="analysis-details" style={{ padding: 12, borderLeft: '1px solid #ddd' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ margin: 0 }}>Analysis Details</h3>
        <div>
          <button onClick={onClose}>Close</button>
        </div>
      </div>
      {loading && <div>Loading...</div>}
      {error && <div style={{ color: 'red' }}>{error}</div>}
      {analysis && (
        <div style={{ marginTop: 8 }}>
          {analysis.error && <div style={{ color: 'red', marginBottom: 8 }}>Error: {analysis.error}</div>}
          <div><strong>Filename:</strong> {analysis.filename}</div>
          <div><strong>Model:</strong> {analysis.model}</div>
          <div><strong>Created:</strong> {analysis.created_at}</div>
          <div style={{ marginTop: 8 }}>
            <strong>Video:</strong>
            {analysis.video_url ? (
              <div style={{ marginTop: 6 }}>
                <a href={`http://localhost:8001${analysis.video_url}`} target="_blank" rel="noreferrer">Open video</a>
                <div style={{ marginTop: 8 }}>
                  <video ref={videoRef} controls width="100%">
                    <source src={`http://localhost:8001${analysis.video_url}`} />
                  </video>

                  <div style={{ display: 'flex', gap: 12, marginTop: 10 }}>
                    <div style={{ flex: 1, textAlign: 'left' }}>
                      <strong>Idle Frames</strong>
                        {analysis.idle_ranges && analysis.idle_ranges.length > 0 ? (
                          <div className="captions-list">
                            {analysis.idle_ranges.slice(0,50).map((r, i) => (
                              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                                <div style={{ flex: 1 }}>
                                  <small>{r.startTime.toFixed(2)}s - {r.endTime.toFixed(2)}s</small>
                                  <div style={{ fontSize: 12, color: '#444' }}>{r.captions && r.captions.length ? r.captions.join(' | ') : ''}</div>
                                </div>
                                <div>
                                  <button onClick={() => playRange(r.startTime, r.endTime)}>Play</button>
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (<div style={{ color: '#666' }}>No idle frames detected.</div>)}
                    </div>

                    <div style={{ flex: 1, textAlign: 'left' }}>
                      <strong>Work Frames</strong>
                      {analysis.work_ranges && analysis.work_ranges.length > 0 ? (
                        <div className="captions-list">
                          {analysis.work_ranges.slice(0,50).map((r, i) => (
                            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                              <div style={{ flex: 1 }}>
                                <small>{r.startTime.toFixed(2)}s - {r.endTime.toFixed(2)}s</small>
                                <div style={{ fontSize: 12, color: '#444' }}>{r.captions && r.captions.length ? r.captions.join(' | ') : ''}</div>
                              </div>
                              <div>
                                <button onClick={() => playRange(r.startTime, r.endTime)}>Play</button>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (<div style={{ color: '#666' }}>No work frames detected.</div>)}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ color: '#666' }}>No video available.</div>
            )}
          </div>

          <div style={{ marginTop: 10 }}>
            <strong>Assignment:</strong>
            {analysis.assignment_id ? (
              <div>
                <div>ID: {analysis.assignment_id}</div>
              </div>
            ) : (
              <div style={{ color: '#666' }}>None</div>
            )}
          </div>

          <div style={{ marginTop: 10 }}>
            <strong>Samples ({(analysis.samples || []).length}):</strong>
            <div style={{ maxHeight: '40vh', overflow: 'auto', marginTop: 6 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th>Frame</th>
                    <th>Time</th>
                    <th>Label</th>
                    <th>Caption</th>
                    <th>LLM</th>
                  </tr>
                </thead>
                <tbody>
                  {(analysis.samples || []).map((s, i) => (
                    <tr key={i} style={{ borderTop: '1px solid #eee' }}>
                      <td style={{ padding: 6 }}>{s.frame_index}</td>
                      <td style={{ padding: 6 }}>{s.time_sec ? s.time_sec.toFixed(2) : ''}</td>
                      <td style={{ padding: 6 }}>{s.label}</td>
                      <td style={{ padding: 6, maxWidth: 300 }}>{s.caption}</td>
                      <td style={{ padding: 6, maxWidth: 200 }}>{s.llm_output}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
