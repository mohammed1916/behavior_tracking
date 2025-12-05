import React, { useState, useRef, useEffect } from 'react';

export default function FileUpload({ accept = 'video/*', onFileSelect, initialFile = null, label = 'Select file' }) {
  const [dragOver, setDragOver] = useState(false);
  const [file, setFile] = useState(initialFile);
  const inputRef = useRef(null);

  useEffect(() => {
    setFile(initialFile);
  }, [initialFile]);

  const handleFiles = (files) => {
    const f = files && files[0] ? files[0] : null;
    setFile(f);
    if (onFileSelect) onFileSelect(f);
  };

  const onInputChange = (e) => handleFiles(e.target.files);

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFiles(e.dataTransfer.files);
  };

  const openFileDialog = () => {
    if (inputRef.current) inputRef.current.click();
  };

  return (
    <div>
      <div
        className={`upload-section ${dragOver ? 'drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={openFileDialog}
        role="button"
        tabIndex={0}
        style={{ cursor: 'pointer' }}
      >
        <input ref={inputRef} type="file" accept={accept} onChange={onInputChange} style={{ display: 'none' }} />
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
          <div style={{ fontWeight: 600 }}>{label}</div>
          <div style={{ color: 'var(--muted)' }}>Drag & drop here or click to choose a file</div>
          {file ? (
            <div style={{ marginTop: 8, width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ textAlign: 'left', overflow: 'hidden', textOverflow: 'ellipsis' }}>{file.name}</div>
                <div>
                  <button onClick={(e) => { e.stopPropagation(); handleFiles(null); }} style={{ marginLeft: 8 }}>Clear</button>
                </div>
              </div>
              {file.type && file.type.startsWith('video') && (
                <video src={URL.createObjectURL(file)} style={{ marginTop: 8, maxWidth: '100%', borderRadius: 8 }} controls />
              )}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
