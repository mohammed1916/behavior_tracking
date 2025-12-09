import React, { useState, useRef, useEffect } from 'react';

export default function LiveView() {
  const [running, setRunning] = useState(false);
  const imgRef = useRef(null);
  const eventSourceRef = useRef(null);

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const startLive = () => {
    const base = `${window.location.protocol}//${window.location.hostname}:8001`;
    const url = base + '/backend/stream_pose';

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.onopen = () => {
      console.log("Stream connected");
      setRunning(true);
    };

    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        const b64 = data.image || data.frame || data.jpeg || data.jpg;
        if (b64 && imgRef.current) {
          imgRef.current.src = `data:image/jpeg;base64,${b64}`;
        }
      } catch {
        if (ev.data?.length > 50 && imgRef.current) {
          imgRef.current.src = `data:image/jpeg;base64,${ev.data}`;
        }
      }
    };

    es.onerror = () => {
      console.warn("Stream lost");
      stopLive();
    };
  };

  const stopLive = () => {
    setRunning(false);

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    if (imgRef.current) imgRef.current.src = '';

    const base = `${window.location.protocol}//${window.location.hostname}:8001`;
    fetch(base + '/backend/stop_webcam', { method: 'POST' }).catch(() => {});
  };

  return (
    <div className="panel">
      <h4>Live View</h4>
      <div style={{ marginTop: 8 }}>
        <div style={{ color: 'var(--muted)', marginBottom: 8 }}>Connect to Webcam or server stream here.</div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <button onClick={running ? stopLive : startLive}>
            {running ? 'Stop' : 'Start'} Live
          </button>
          <div style={{ color: 'var(--muted)' }}>
            {running ? 'Status: Running...' : 'Status: Not Running'}
          </div>
        </div>
        <div style={{ marginTop: 12 }}>
          {running ? (
            <img ref={imgRef} style={{ width: '100%', borderRadius: 8 }} alt="Live stream" />
          ) : (
            <div style={{ padding: 24, border: '1px dashed var(--panel-border)', borderRadius: 8, color: 'var(--muted)' }}>
              Live view is stopped.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
