import React, { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showPose, setShowPose] = useState(false);
  const [mode, setMode] = useState('upload');

  // VLM state
  const [vlmModel, setVlmModel] = useState('gpt-4o-vlm');
  const [vlmPrompt, setVlmPrompt] = useState('');
  const [vlmVideo, setVlmVideo] = useState(null);
  const [vlmResult, setVlmResult] = useState(null);
  const [vlmLoading, setVlmLoading] = useState(false);
  const [vlmUseLocal, setVlmUseLocal] = useState(true);

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

      const endpoint = vlmUseLocal ? 'http://localhost:8001/backend/vlm_local' : 'http://localhost:8001/backend/vlm';
      const resp = await fetch(endpoint, { method: 'POST', body: formData });
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
    if (!file) return alert('Please select a file first!');
    setLoading(true);
    setStatus('Uploading and processing...');
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await fetch('http://localhost:8001/backend/analyze_video', { method: 'POST', body: formData });
      if (!response.ok) throw new Error(`Error: ${response.statusText}`);
      const data = await response.json();
      setResult(data);
      setStatus('Processing complete!');
    } catch (error) {
      console.error('Error uploading file:', error);
      setStatus(`Failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Behavior Tracking Analysis</h1>

      <div className="controls">
        <div className="panel left">
          <div style={{ marginBottom: 12 }} className="mode-toggle">
            <input id="mode-upload" type="radio" name="mode" value="upload" checked={mode === 'upload'} onChange={() => { setMode('upload'); setShowPose(false); }} />
            <label htmlFor="mode-upload">Upload</label>
            <input id="mode-stream" type="radio" name="mode" value="stream" checked={mode === 'stream'} onChange={() => { setMode('stream'); setShowPose(false); }} />
            <label htmlFor="mode-stream">Live</label>
          </div>

          {mode === 'upload' && (
            <div className="upload-section">
              <input type="file" accept="video/*" onChange={handleFileChange} />
              <button onClick={handleUpload} disabled={loading || !file}>{loading ? 'Processing...' : 'Analyze Video'}</button>
            </div>
          )}

          {mode === 'stream' && (
            <div className="stream-section">
              <div style={{ marginBottom: 8 }}>
                <button onClick={() => setShowPose((s) => !s)}>{showPose ? 'Stop Stream' : 'Start Stream'}</button>
              </div>
              {showPose && (
                <div className="pose-stream">
                  <h3>Live Pose</h3>
                  <img src="http://localhost:8001/backend/stream_pose" alt="Live Pose" style={{ maxWidth: '100%' }} onError={() => setShowPose(false)} />
                </div>
              )}
            </div>
          )}

          <div style={{ marginTop: 14 }} className="panel">
            <h3>Results</h3>
            {status && <p className="status">{status}</p>}
            {result ? (
              <>
                <p><strong>Task Completed:</strong> {result.task_completed ? 'YES' : 'NO'}</p>
                {result.download_url && (
                  <div className="video-container">
                    <h4>Processed Video</h4>
                    <video controls width="100%"><source src={`http://localhost:8001${result.download_url}`} type="video/mp4" /></video>
                    <div style={{ marginTop: 8 }}><a href={`http://localhost:8001${result.download_url}`} download><button>Download Processed Video</button></a></div>
                  </div>
                )}
              </>
            ) : (
              <p>No results yet.</p>
            )}
          </div>
        </div>

        <div className="panel right">
          <h3>VLM (Video)</h3>
          <div className="vlm-section">
            <label>Model:
              <select value={vlmModel} onChange={(e) => setVlmModel(e.target.value)}>
                <option value="gpt-4o-vlm">gpt-4o-vlm</option>
                <option value="gpt-5-vlm">gpt-5-vlm</option>
                <option value="openai-vision">openai-vision</option>
              </select>
            </label>

            <label>Prompt:
              <textarea value={vlmPrompt} onChange={(e) => setVlmPrompt(e.target.value)} placeholder="Ask about the video or request an analysis" rows={3} />
            </label>

            <label>Optional Video:
              <input type="file" accept="video/*" onChange={handleVlmVideoChange} />
            </label>

            <label style={{ display: 'block', marginTop: 8 }}>
              <input type="checkbox" checked={vlmUseLocal} onChange={(e) => setVlmUseLocal(e.target.checked)} /> Use local VLM
            </label>

            <div style={{ marginTop: 8 }}>
              <button onClick={handleVlmSubmit} disabled={vlmLoading}>{vlmLoading ? 'Running...' : 'Run VLM on Video'}</button>
            </div>

            {vlmResult && (
              <div className="vlm-result" style={{ marginTop: 12 }}>
                <h4>VLM Result</h4>
                {vlmResult.error ? (
                  <pre style={{ color: 'red' }}>{vlmResult.error}</pre>
                ) : (
                  <>
                    <div style={{ marginBottom: 8 }}><strong>Analysis</strong><pre className="vlm-result">{JSON.stringify(vlmResult.analysis, null, 2)}</pre></div>
                    {vlmResult.analysis && (vlmResult.analysis.captions || vlmResult.analysis.caption) && (
                      <div><strong>Captions</strong><ul className="captions-list">{vlmResult.analysis.captions ? vlmResult.analysis.captions.map((c, i) => <li key={i}>{c}</li>) : <li>{vlmResult.analysis.caption}</li>}</ul></div>
                    )}
                    {vlmResult.analysis && vlmResult.analysis.video_url && (
                      <div style={{ marginTop: 8 }}><strong>Preview</strong><video controls width="100%" style={{ marginTop: 8 }}><source src={`http://localhost:8001${vlmResult.analysis.video_url}`} type="video/mp4" /></video></div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;

