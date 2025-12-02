import React, { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setStatus('');
    setResult(null);
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

      <div className="upload-section">
        <input type="file" accept="video/*" onChange={handleFileChange} />
        <button onClick={handleUpload} disabled={loading || !file}>
          {loading ? 'Processing...' : 'Analyze Video'}
        </button>
      </div>

      {status && <p className="status">{status}</p>}

      {result && (
        <div className="result-section">
          <h2>Results</h2>
          <p><strong>Task Completed:</strong> {result.task_completed ? "YES" : "NO"}</p>

          {result.download_url && (
            <div className="video-container">
              <h3>Processed Video</h3>
              <video controls width="640">
                <source src={`http://localhost:8001${result.download_url}`} type="video/mp4" />
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
    </div>
  );
}

export default App;
