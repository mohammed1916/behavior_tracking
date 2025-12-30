import React, { useEffect, useState } from 'react';

export default function StoredAnalyses({ onView }) {
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchList = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch('http://localhost:8001/backend/analyses');
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      setAnalyses(data || []);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchList(); }, []);

  const handleDelete = async (id) => {
    if (!window.confirm('Delete analysis ' + id + '?')) return;
    try {
      const resp = await fetch(`http://localhost:8001/backend/analysis/${encodeURIComponent(id)}`, { method: 'DELETE' });
      if (!resp.ok) throw new Error(await resp.text());
      setAnalyses((a) => a.filter(x => x.id !== id));
    } catch (e) {
      alert('Delete failed: ' + (e.message || String(e)));
    }
  };

  return (
    <div className="stored-analyses">
      <h3>Stored Analyses</h3>
      <div style={{ marginBottom: 8 }}>
        <button onClick={fetchList} disabled={loading}>{loading ? 'Refreshing...' : 'Refresh'}</button>
      </div>
      {error && <div style={{ color: 'red' }}>{error}</div>}
      {!loading && analyses.length === 0 && <div style={{ color: 'var(--muted)' }}>No stored analyses found.</div>}
      {analyses.length > 0 && (
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left' }}>Filename</th>
              <th>Model</th>
              <th>Work / Idle</th>
              <th>Created</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {analyses.map(a => (
              <tr key={a.id} style={{ borderTop: '1px solid var(--panel-border)' }}>
                <td style={{ padding: '6px 4px' }}>{a.filename}</td>
                <td style={{ textAlign: 'center' }}>{a.model}</td>
                <td style={{ textAlign: 'center', fontSize: 12 }}>
                  <span style={{ color: 'rgba(76, 175, 80, 1)', fontWeight: 'bold' }}>
                    {(a.work_percentage || 0).toFixed(1)}%
                  </span>
                  {' / '}
                  <span style={{ color: 'rgba(158, 158, 158, 1)', fontWeight: 'bold' }}>
                    {(a.idle_percentage || 0).toFixed(1)}%
                  </span>
                </td>
                <td style={{ textAlign: 'center' }}>{a.created_at ? new Date(a.created_at).toLocaleString() : ''}</td>
                <td style={{ textAlign: 'center' }}>
                  <button onClick={() => onView && onView(a.id)} style={{ marginRight: 8 }}>View</button>
                  <button onClick={() => handleDelete(a.id)}>Delete</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
