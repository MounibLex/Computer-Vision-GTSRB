from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
model = YOLO('best.pt')  # make sure best.pt is in the same folder

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Traffic Sign Detector</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0a0a0f;
    --surface: #13131a;
    --border: #2a2a3a;
    --accent: #e8ff47;
    --accent2: #ff4747;
    --text: #f0f0f0;
    --muted: #666680;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 48px 24px;
  }

  header {
    text-align: center;
    margin-bottom: 48px;
  }

  .tag {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.15em;
    color: var(--accent);
    border: 1px solid var(--accent);
    padding: 4px 12px;
    margin-bottom: 16px;
    text-transform: uppercase;
  }

  h1 {
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.02em;
  }

  h1 span { color: var(--accent); }

  .subtitle {
    margin-top: 12px;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 13px;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    width: 100%;
    max-width: 760px;
    padding: 40px;
  }

  .drop-zone {
    border: 2px dashed var(--border);
    padding: 60px 24px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    position: relative;
  }

  .drop-zone:hover, .drop-zone.drag { border-color: var(--accent); background: rgba(232,255,71,0.03); }

  .drop-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }

  .drop-icon {
    font-size: 48px;
    margin-bottom: 16px;
    display: block;
  }

  .drop-label {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 8px;
  }

  .drop-sub {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: var(--muted);
  }

  .conf-row {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-top: 24px;
  }

  .conf-row label {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: var(--muted);
    white-space: nowrap;
  }

  input[type=range] {
    flex: 1;
    accent-color: var(--accent);
    height: 4px;
  }

  #conf-val {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    color: var(--accent);
    min-width: 36px;
  }

  .btn {
    margin-top: 24px;
    width: 100%;
    padding: 16px;
    background: var(--accent);
    color: #0a0a0f;
    border: none;
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    cursor: pointer;
    transition: opacity 0.2s;
  }

  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn:hover:not(:disabled) { opacity: 0.85; }

  .results {
    margin-top: 32px;
    display: none;
  }

  .results.show { display: block; }

  .results-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
  }

  .results-title {
    font-size: 13px;
    font-family: 'DM Mono', monospace;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .det-count {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: var(--accent);
    border: 1px solid var(--accent);
    padding: 2px 10px;
  }

  #result-img {
    width: 100%;
    display: block;
    border: 1px solid var(--border);
  }

  .detections {
    margin-top: 16px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .det-tag {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    padding: 4px 10px;
    border: 1px solid var(--border);
    color: var(--text);
    background: var(--bg);
  }

  .det-tag span { color: var(--accent); }

  .loading {
    display: none;
    text-align: center;
    padding: 32px;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    color: var(--muted);
  }

  .loading.show { display: block; }

  .spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-right: 10px;
    vertical-align: middle;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  .no-det {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    color: var(--accent2);
    padding: 12px;
    border: 1px solid var(--accent2);
    margin-top: 16px;
  }

  .stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: var(--border);
    margin-top: 16px;
  }

  .stat {
    background: var(--bg);
    padding: 16px;
    text-align: center;
  }

  .stat-val {
    font-size: 22px;
    font-weight: 800;
    color: var(--accent);
  }

  .stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    margin-top: 4px;
    text-transform: uppercase;
  }
</style>
</head>
<body>

<header>
  <div class="tag">YOLOv8 · 43 Classes · mAP50 0.995</div>
  <h1>Traffic Sign<br><span>Detector</span></h1>
  <p class="subtitle">GTSRB — German Traffic Sign Recognition Benchmark</p>
</header>

<div class="card">
  <div class="drop-zone" id="drop-zone">
    <input type="file" id="file-input" accept="image/*">
    <span class="drop-icon">🚦</span>
    <div class="drop-label">Drop an image here</div>
    <div class="drop-sub">or click to browse · JPG, PNG, WEBP</div>
  </div>

  <div class="conf-row">
    <label>CONFIDENCE</label>
    <input type="range" id="conf" min="5" max="95" value="25">
    <span id="conf-val">0.25</span>
  </div>

  <button class="btn" id="detect-btn" disabled>Detect Signs</button>

  <div class="loading" id="loading">
    <span class="spinner"></span> Running inference...
  </div>

  <div class="results" id="results">
    <div class="results-header">
      <span class="results-title">Detection Result</span>
      <span class="det-count" id="det-count">0 detections</span>
    </div>
    <img id="result-img" src="" alt="Result">
    <div class="stats-row">
      <div class="stat">
        <div class="stat-val" id="stat-det">—</div>
        <div class="stat-label">Detections</div>
      </div>
      <div class="stat">
        <div class="stat-val" id="stat-conf">—</div>
        <div class="stat-label">Avg Confidence</div>
      </div>
      <div class="stat">
        <div class="stat-val" id="stat-time">—</div>
        <div class="stat-label">Inference ms</div>
      </div>
    </div>
    <div class="detections" id="detections"></div>
  </div>
</div>

<script>
  const fileInput = document.getElementById('file-input');
  const dropZone = document.getElementById('drop-zone');
  const detectBtn = document.getElementById('detect-btn');
  const confSlider = document.getElementById('conf');
  const confVal = document.getElementById('conf-val');
  const loading = document.getElementById('loading');
  const results = document.getElementById('results');
  const resultImg = document.getElementById('result-img');
  const detCount = document.getElementById('det-count');
  const detections = document.getElementById('detections');

  let selectedFile = null;

  confSlider.oninput = () => confVal.textContent = (confSlider.value / 100).toFixed(2);

  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  fileInput.onchange = () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); };

  function handleFile(file) {
    selectedFile = file;
    dropZone.querySelector('.drop-label').textContent = file.name;
    dropZone.querySelector('.drop-sub').textContent = (file.size / 1024).toFixed(0) + ' KB';
    dropZone.querySelector('.drop-icon').textContent = '🖼️';
    detectBtn.disabled = false;
    results.classList.remove('show');
  }

  detectBtn.onclick = async () => {
    if (!selectedFile) return;
    loading.classList.add('show');
    results.classList.remove('show');
    detectBtn.disabled = true;

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('conf', confSlider.value / 100);

    const start = Date.now();
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();
    const elapsed = Date.now() - start;

    loading.classList.remove('show');
    detectBtn.disabled = false;

    resultImg.src = 'data:image/jpeg;base64,' + data.image;
    detCount.textContent = data.detections.length + ' detection' + (data.detections.length !== 1 ? 's' : '');

    document.getElementById('stat-det').textContent = data.detections.length;
    document.getElementById('stat-conf').textContent = data.detections.length
      ? (data.detections.reduce((s, d) => s + d.conf, 0) / data.detections.length * 100).toFixed(0) + '%'
      : '—';
    document.getElementById('stat-time').textContent = elapsed + 'ms';

    detections.innerHTML = data.detections.map(d =>
      `<div class="det-tag">${d.label} <span>${(d.conf * 100).toFixed(0)}%</span></div>`
    ).join('');

    results.classList.add('show');
  };
</script>
</body>
</html>'''

@app.route('/')
def index():
    return HTML

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    conf = float(request.form.get('conf', 0.25))
    
    img = Image.open(file.stream).convert('RGB')
    results = model.predict(img, conf=conf, verbose=False)
    
    # Draw results
    result_img = results[0].plot()
    result_pil = Image.fromarray(result_img[..., ::-1])
    
    buf = io.BytesIO()
    result_pil.save(buf, format='JPEG', quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    
    # Extract detections
    boxes = results[0].boxes
    names = model.names
    dets = []
    if boxes is not None:
        for box in boxes:
            dets.append({
                'label': names[int(box.cls)],
                'conf': float(box.conf)
            })
    
    return jsonify({'image': img_b64, 'detections': dets})

if __name__ == '__main__':
    app.run(debug=True, port=5000)