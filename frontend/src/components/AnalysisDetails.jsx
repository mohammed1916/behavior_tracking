import React, { useEffect, useState } from 'react';

export default function AnalysisDetails({ analysisId, onClose }) {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

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
                  <video controls width="100%">
                    <source src={`http://localhost:8001${analysis.video_url}`} type="video/mp4" />
                  </video>
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
