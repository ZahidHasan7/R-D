const { useState, useRef } = React;

function App() {
  const [asrText, setAsrText] = useState("");
  const [ttsText, setTtsText] = useState("Hello from Voice Agent");
  const audioRef = useRef(null);
  const fileRef = useRef(null);

  async function uploadASR() {
    const f = fileRef.current.files[0];
    if (!f) return alert("Choose an audio file");
    const fd = new FormData();
    fd.append("file", f, f.name);
    const res = await fetch("/asr", { method: "POST", body: fd });
    if (!res.ok) return alert("ASR failed: " + res.statusText);
    const j = await res.json();
    setAsrText(j.text || "");
  }

  async function synthesize() {
    const res = await fetch("/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: ttsText }),
    });
    if (!res.ok) return alert("TTS failed: " + res.statusText);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    if (audioRef.current) audioRef.current.src = url;
    audioRef.current.play();
  }

  async function agentRun() {
    const f = fileRef.current.files[0];
    if (!f) return alert("Choose an audio file");
    const fd = new FormData();
    fd.append("file", f, f.name);
    const res = await fetch("/agent", { method: "POST", body: fd });
    if (!res.ok) return alert("Agent failed: " + res.statusText);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    if (audioRef.current) audioRef.current.src = url;
    audioRef.current.play();
  }

  return (
    <div>
      <h2>Voice Agent UI</h2>

      <section>
        <h3>ASR (upload audio)</h3>
        <input type="file" ref={fileRef} accept="audio/*" />
        <div>
          <button onClick={uploadASR}>Transcribe</button>
          <button onClick={agentRun}>Send to Agent (ASR → Reply)</button>
        </div>
        <div>
          <label>Transcript:</label>
          <textarea value={asrText} readOnly />
        </div>
      </section>

      <hr />

      <section>
        <h3>TTS (text → audio)</h3>
        <textarea
          value={ttsText}
          onChange={(e) => setTtsText(e.target.value)}
        />
        <div>
          <button onClick={synthesize}>Synthesize</button>
        </div>
        <audio
          ref={audioRef}
          controls
          style={{ width: "100%", marginTop: 8 }}
        />
      </section>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
