import React, { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showPose, setShowPose] = useState(false);
  const [mode, setMode] = useState('upload'); // 'upload' or 'stream'
  // VLM (Visual Language Model) UI state
  const [vlmModel, setVlmModel] = useState('gpt-4o-vlm');
  const [vlmPrompt, setVlmPrompt] = useState('');
  const [vlmVideo, setVlmVideo] = useState(null);
  const [vlmResult, setVlmResult] = useState(null);
  const [vlmLoading, setVlmLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setStatus('');
    setResult(null);
  };

  const handleVlmVideoChange = (e) => {
    setVlmVideo(e.target.files[0] || null);
    setVlmResult(null);
  };

  const handleVlmSubmit = async () => {
    if (!vlmPrompt && !vlmVideo) {
      alert('Enter a prompt or select a video for the VLM.');
      return;
    }

    setVlmLoading(true);
    setVlmResult(null);

    try {
      const formData = new FormData();
      formData.append('model', vlmModel);
      formData.append('prompt', vlmPrompt);
      if (vlmVideo) formData.append('video', vlmVideo);

      const resp = await fetch('http://localhost:8001/backend/vlm', {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(txt || resp.statusText);
      }

      const data = await resp.json();
      setVlmResult(data);
    } catch (err) {
      console.error('VLM request failed', err);
      setVlmResult({ error: err.message });
    } finally {
      setVlmLoading(false);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first!");
      return;
    }

    setLoading(true);
    setStatus('Uploading and processing...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8001/backend/analyze_video', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
      setStatus('Processing complete!');
    } catch (error) {
      console.error("Error uploading file:", error);
      setStatus(`Failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Behavior Tracking Analysis</h1>

      <div className="mode-toggle" style={{ marginBottom: 12 }}>
        <label style={{ marginRight: 12 }}>
          <input type="radio" name="mode" value="upload" checked={mode === 'upload'} onChange={() => { setMode('upload'); setShowPose(false); }} /> Upload Video
        </label>
        <label>
          <input type="radio" name="mode" value="stream" checked={mode === 'stream'} onChange={() => { setMode('stream'); setShowPose(false); }} /> Live Stream
        </label>
      </div>

      {mode === 'upload' && (
        <div className="upload-section">
          <input type="file" accept="video/*" onChange={handleFileChange} />
          <button onClick={handleUpload} disabled={loading || !file}>
            {loading ? 'Processing...' : 'Analyze Video'}
          </button>
        </div>
      )}

      {mode === 'stream' && (
        <div className="stream-section">
          <div style={{ marginBottom: 8 }}>
            <button onClick={() => setShowPose((s) => !s)}>{showPose ? 'Stop Stream' : 'Start Stream'}</button>
          </div>
          {showPose && (
            <div className="pose-stream">
              <h2>Live Pose</h2>
              <img
                src="http://localhost:8001/backend/stream_pose"
                alt="Live Pose"
                style={{ maxWidth: '100%' }}
                onError={() => setShowPose(false)}
              />
            </div>
          )}
        </div>
      )}

      {status && <p className="status">{status}</p>}

      {result && (
        <div className="result-section">
          <h2>Results</h2>
          <p><strong>Task Completed:</strong> {result.task_completed ? "YES" : "NO"}</p>

          {result.download_url && (
            <div className="video-container">
              <h3>Processed Video</h3>
              <video controls width="640">
                <source src={`http://localhost:8001/${result.download_url}`} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
              <br />
              <a href={`http://localhost:8001${result.download_url}`} download>
                <button>Download Processed Video</button>
              </a>
            </div>
          )}
        </div>
      )}

      {/* VLM (Visual Language Model) Section - moved below results */}
      <div className="vlm-section" style={{ marginTop: 20 }}>
        <h2>Visual Language Model (Video)</h2>

        <label>
          Model:
          <select value={vlmModel} onChange={(e) => setVlmModel(e.target.value)}>
            <option value="gpt-4o-vlm">gpt-4o-vlm</option>
            <option value="gpt-5-vlm">gpt-5-vlm</option>
            <option value="openai-vision">openai-vision</option>
          </select>
        </label>

        <label>
          Prompt:
          <textarea
            value={vlmPrompt}
            onChange={(e) => setVlmPrompt(e.target.value)}
            placeholder="Ask the model about the video or request an analysis/summary"
            rows={3}
            style={{ width: '100%' }}
          />
        </label>

        <label>
          Optional Video:
          <input type="file" accept="video/*" onChange={handleVlmVideoChange} />
        </label>

        <div>
          <button onClick={handleVlmSubmit} disabled={vlmLoading}>
            {vlmLoading ? 'Running...' : 'Run VLM on Video'}
          </button>
        </div>

        {vlmResult && (
          <div className="vlm-result">
            <h3>VLM Result</h3>
            {vlmResult.error ? (
              <pre style={{ color: 'red' }}>{vlmResult.error}</pre>
            ) : (
              <pre>{JSON.stringify(vlmResult, null, 2)}</pre>
            )}
          </div>
        )}
      </div>
      {/* (Removed duplicate live/result blocks) */}
    </div>
  );
}

export default App;
