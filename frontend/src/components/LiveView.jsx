import React, { useState, useRef } from 'react';

export default function LiveView() {
  const [running, setRunning] = useState(false);
  const videoRef = useRef(null);

  const startLive = () => {
    setRunning(true);
  };

  const stopLive = () => {
    setRunning(false);
    try { if (videoRef.current) videoRef.current.pause(); } catch {}
    // Stop webcam on backend
    fetch('http://localhost:8001/backend/stop_webcam', { method: 'POST' }).catch(console.error);
  };

  return (
    <div className="panel">
      <h4>Live View</h4>
      <div style={{ marginTop: 8 }}>
        <div style={{ color: 'var(--muted)', marginBottom: 8 }}> Connect to Webcam or server stream here.</div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <button onClick={running ? stopLive : startLive}>{running ? 'Stop' : 'Start'} Live</button>
          <div style={{ color: 'var(--muted)' }}>{running ? 'Status: Running...' : 'Status: Not Running'}</div>
        </div>
        <div style={{ marginTop: 12 }}>
          {running ? (
            <img ref={videoRef} src="http://localhost:8001/backend/stream_pose" style={{ width: '100%', borderRadius: 8 }} alt="Live stream" />
          ) : (
            <div style={{ padding: 24, border: '1px dashed var(--panel-border)', borderRadius: 8, color: 'var(--muted)' }}>Live view is stopped.</div>
          )}
        </div>
      </div>
    </div>
  );
}
