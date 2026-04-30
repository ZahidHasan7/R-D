const { useState, useRef } = React;

function App() {
  const [asrText, setAsrText] = useState("");
  const [ttsText, setTtsText] = useState("Hello from Voice Agent");
  const [isLoading, setIsLoading] = useState(false);
  const audioRef = useRef(null);
  const fileRef = useRef(null);

  // Robust audio playback helper
  const playAudio = (url) => {
    if (!audioRef.current) return;
    
    // Pre-emptive pause
    audioRef.current.pause();
    audioRef.current.src = url;
    
    // In some browsers, explicitly calling load() helps reset the state machine
    audioRef.current.load();
    
    const playPromise = audioRef.current.play();
    if (playPromise !== undefined) {
      playPromise.catch(error => {
        if (error.name === 'AbortError') {
          console.log('Playback was interrupted by a newer request or state change.');
        } else {
          console.error('Audio playback error:', error);
        }
      });
    }
  };

  async function uploadASR() {
    if (isLoading) return;
    const f = fileRef.current.files[0];
    if (!f) return alert("Choose an audio file");
    
    setIsLoading(true);
    try {
      const fd = new FormData();
      fd.append("file", f, f.name);
      const res = await fetch("/asr", { method: "POST", body: fd });
      if (!res.ok) {
        alert("ASR failed: " + res.statusText);
        return;
      }
      const j = await res.json();
      setAsrText(j.text || "");
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      setIsLoading(false);
    }
  }

  async function synthesize() {
    if (isLoading || !ttsText.trim()) return;
    setIsLoading(true);
    try {
      const res = await fetch("/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: ttsText }),
      });
      if (!res.ok) {
        alert("TTS failed: " + res.statusText);
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      playAudio(url);
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      setIsLoading(false);
    }
  }

  async function agentRun() {
    if (isLoading) return;
    const f = fileRef.current.files[0];
    if (!f) return alert("Choose an audio file");
    
    setIsLoading(true);
    try {
      const fd = new FormData();
      fd.append("file", f, f.name);
      const res = await fetch("/agent", { method: "POST", body: fd });
      if (!res.ok) {
        alert("Agent failed: " + res.statusText);
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      playAudio(url);
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div>
      <h2>Bangla Voice Agent {isLoading && <span className="loading-indicator">Processing Request</span>}</h2>

      <section style={{ opacity: isLoading ? 0.6 : 1 }}>
        <h3>Speech Recognition (ASR)</h3>
        <input type="file" ref={fileRef} accept="audio/*" disabled={isLoading} />
        <div>
          <button onClick={uploadASR} disabled={isLoading}>
            {isLoading ? 'Wait...' : 'Transcribe'}
          </button>
          <button onClick={agentRun} disabled={isLoading}>
            {isLoading ? 'Wait...' : 'Send to Agent'}
          </button>
        </div>
        <div>
          <label>Transcript Output</label>
          <textarea value={asrText} readOnly disabled={isLoading} placeholder="Transcription results will appear here..." />
        </div>
      </section>

      <section style={{ opacity: isLoading ? 0.6 : 1 }}>
        <h3>Text-to-Speech (TTS)</h3>
        <textarea
          value={ttsText}
          onChange={(e) => setTtsText(e.target.value)}
          disabled={isLoading}
          placeholder="Enter text to synthesize..."
        />
        <div>
          <button onClick={synthesize} disabled={isLoading}>
            {isLoading ? 'Wait...' : 'Synthesize Audio'}
          </button>
        </div>
        <audio
          ref={audioRef}
          controls
          style={{ width: "100%", marginTop: 8 }}
        />
      </section>

      <div className="version-tag">v2.1.0 - Enterprise Edition</div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
