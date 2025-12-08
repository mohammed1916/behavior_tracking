import React, { useState, useRef } from 'react';

export default function LiveView() {
  const [running, setRunning] = useState(false);
  const videoRef = useRef(null);

  const startLive = () => {
    // Placeholder: in future connect to real stream endpoint
    setRunning(true);
  };

  const stopLive = () => {
    setRunning(false);
    try { if (videoRef.current) videoRef.current.pause(); } catch {}
  };

  return (
    <div className="panel">
      <h4>Live View</h4>
      <div style={{ marginTop: 8 }}>
        <div style={{ color: 'var(--muted)', marginBottom: 8 }}> Connect to Webcam or server stream here.</div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <button onClick={running ? stopLive : startLive}>{running ? 'Stop' : 'Start'} Live</button>
          <div style={{ color: 'var(--muted)' }}>{running ? 'Running...' : 'Stopped'}</div>
        </div>
        <div style={{ marginTop: 12 }}>
          {running ? (
            <video ref={videoRef} style={{ width: '100%', borderRadius: 8 }} autoPlay muted playsInline>
              {/* In a real setup we could set a source here, e.g. an HLS or MJPEG stream */}
            </video>
          ) : (
            <div style={{ padding: 24, border: '1px dashed var(--panel-border)', borderRadius: 8, color: 'var(--muted)' }}>Live view is stopped.</div>
          )}
        </div>
      </div>
    </div>
  );
}
