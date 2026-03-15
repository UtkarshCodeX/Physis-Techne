"""
Flask backend — supports two models:
  Model A: engine.py      (Fock-space, 3 SPDC sources, sparse matrices)
  Model B: alok_final.py  (State-vector, 2 SPDC sources, genetic search)

Routes
------
  GET  /              → UI
  GET  /stream        → SSE log stream
  GET  /download_pdf  → PDF report of last result
  GET  /last_result   → JSON debug dump
"""

import json
import queue
import threading
import time
import sys
import os
import io
from pathlib import Path
from io import BytesIO

from flask import Flask, Response, jsonify, request, send_file, render_template_string, stream_with_context

# ── Import both engines ────────────────────────────────────────────────────────

_here = Path(__file__).parent
for _candidate in [_here, _here / 'outputs', Path('.')]:
    sys.path.insert(0, str(_candidate))

# Engine A — engine.py
try:
    from engine import (
        run_experiment as run_engine,
        parse_custom_vec, DIM,
        SPDC_BASIS_IDX, SPDC_BASIS_LABELS,
    )
    ENGINE_A_OK = True
except ImportError as _e:
    ENGINE_A_OK = False
    _ENGINE_A_ERR = str(_e)

# Engine B — alok_final.py
try:
    import alok_final as _alok
    ENGINE_B_OK = True
except ImportError as _e:
    ENGINE_B_OK = False
    _ENGINE_B_ERR = str(_e)

# ── Flask app ──────────────────────────────────────────────────────────────────

app = Flask(__name__)
_last_result: dict = {}

# ── HTML UI ───────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Photonic Circuit Optimizer</title>
    <style>
        *, *::before, *::after { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0d1117; color: #c9d1d9;
            margin: 0; padding: 30px; min-height: 100vh;
        }
        .container {
            max-width: 980px; margin: 0 auto;
            background: #161b22; padding: 36px;
            border-radius: 12px; border: 1px solid #30363d;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }
        h1 { margin: 0 0 4px 0; font-size: 1.7rem; color: #e6edf3; }
        .subtitle { font-size: 0.9rem; color: #8b949e; margin-bottom: 28px; }

        /* Model selector banner */
        .model-banner {
            display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap;
        }
        .model-card {
            flex: 1; min-width: 220px;
            border: 2px solid #30363d; border-radius: 10px;
            padding: 14px 18px; cursor: pointer;
            transition: border-color 0.2s, background 0.2s;
            background: #0d1117;
        }
        .model-card:hover { border-color: #58a6ff; background: #0d1d3a; }
        .model-card.selected { border-color: #238636; background: #0d2d0e; }
        .model-card .model-title {
            font-weight: 700; font-size: 1rem; color: #e6edf3; margin-bottom: 4px;
        }
        .model-card .model-tag {
            display: inline-block; font-size: 0.72rem; font-weight: 600;
            padding: 2px 8px; border-radius: 20px; margin-bottom: 6px;
        }
        .model-card.selected .model-tag { background:#238636; color:#fff; }
        .model-card:not(.selected) .model-tag { background:#30363d; color:#8b949e; }
        .model-card .model-desc { font-size: 0.82rem; color: #8b949e; line-height: 1.5; }
        .model-card .model-badge {
            display: inline-block; font-size: 0.72rem;
            padding: 2px 7px; border-radius: 10px; margin-top: 6px;
            background: #21262d; color: #8b949e; border: 1px solid #30363d;
        }

        .controls {
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 24px; margin-bottom: 24px;
        }
        @media(max-width:600px){ .controls{ grid-template-columns:1fr; } }
        .control-group { display: flex; flex-direction: column; gap: 8px; }
        label { font-weight:600; font-size:0.85rem; color:#8b949e;
                text-transform:uppercase; letter-spacing:0.05em; }
        select, input[type=text], input[type=number] {
            padding: 10px 14px; background: #0d1117;
            border: 1px solid #30363d; border-radius: 6px;
            color: #e6edf3; font-size: 0.95rem; transition: border-color 0.2s;
        }
        select:focus, input:focus { outline:none; border-color:#58a6ff; }
        .param-row { display:flex; gap:10px; }
        .param-row input { width:50%; }
        .hint { font-size:0.8rem; color:#6e7681; margin-top:2px; }

        #customPanel {
            display:none; background:#0d1117;
            border:1px solid #30363d; border-radius:8px;
            padding:14px 16px; margin-top:12px;
        }
        #customPanel.visible { display:block; }
        #customPanel label { color:#58a6ff; }
        #customPanel textarea {
            width:100%; background:#161b22;
            border:1px solid #30363d; border-radius:6px;
            color:#e6edf3; font-family:'Courier New',monospace;
            font-size:0.85rem; padding:10px; resize:vertical;
            min-height:80px; margin-top:6px;
        }
        #customPanel .examples { font-size:0.78rem; color:#6e7681; margin-top:6px; }

        /* Alok custom target panel */
        #alokCustomPanel {
            display:none; background:#0d1117;
            border:1px solid #30363d; border-radius:8px;
            padding:14px 16px; margin-top:12px;
        }
        #alokCustomPanel.visible { display:block; }
        #alokCustomPanel label { color:#58a6ff; }
        #alokCustomPanel textarea {
            width:100%; background:#161b22;
            border:1px solid #30363d; border-radius:6px;
            color:#e6edf3; font-family:'Courier New',monospace;
            font-size:0.85rem; padding:10px; resize:vertical;
            min-height:60px; margin-top:6px;
        }
        #alokCustomPanel .examples { font-size:0.78rem; color:#6e7681; margin-top:6px; }

        /* Alok input state toggle */
        #alokInputPanel {
            display:none; background:#0d1117;
            border:1px solid #30363d; border-radius:8px;
            padding:14px 16px; margin-top:12px;
        }
        #alokInputPanel.visible { display:block; }
        #alokInputPanel label { color:#f0883e; }
        #alokInputPanel textarea {
            width:100%; background:#161b22;
            border:1px solid #30363d; border-radius:6px;
            color:#e6edf3; font-family:'Courier New',monospace;
            font-size:0.85rem; padding:10px; resize:vertical;
            min-height:60px; margin-top:6px;
        }

        .btn-row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
        button {
            padding:11px 22px; font-size:0.95rem; font-weight:600;
            border:none; border-radius:6px; cursor:pointer;
            transition:background 0.2s, transform 0.1s;
        }
        button:active { transform:scale(0.98); }
        #runBtn { background:#238636; color:#fff; }
        #runBtn:hover { background:#2ea043; }
        #runBtn:disabled { background:#3d444d; color:#6e7681; cursor:not-allowed; transform:none; }
        #downloadPdfBtn  { background:#1f6feb; color:#fff; }
        #downloadPdfBtn:hover  { background:#388bfd; }
        #downloadPdfBtn:disabled  { background:#3d444d; color:#6e7681; cursor:not-allowed; }
        #downloadJsonBtn { background:#6f42c1; color:#fff; }
        #downloadJsonBtn:hover { background:#8a63d2; }
        #downloadJsonBtn:disabled { background:#3d444d; color:#6e7681; cursor:not-allowed; }
        #downloadLogBtn  { background:#0f6674; color:#fff; }
        #downloadLogBtn:hover  { background:#138496; }
        #downloadLogBtn:disabled  { background:#3d444d; color:#6e7681; cursor:not-allowed; }
        #clearBtn { background:transparent; color:#8b949e; border:1px solid #30363d; }
        #clearBtn:hover { background:#21262d; color:#c9d1d9; }

        #statusBadge {
            display:inline-flex; align-items:center; gap:6px;
            font-size:0.82rem; padding:4px 10px; border-radius:20px;
            background:#21262d; color:#8b949e; border:1px solid #30363d;
        }
        .dot { width:8px; height:8px; border-radius:50%; background:currentColor; }
        .dot.pulse { animation:pulse 1.2s infinite; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

        .progress-wrap { height:3px; background:#21262d; border-radius:2px; margin:16px 0 0 0; overflow:hidden; }
        #progressBar { height:100%; background:#238636; width:0; transition:width 0.5s; border-radius:2px; }
        #progressBar.indeterminate { animation:indeterminate 1.5s infinite ease-in-out; }
        @keyframes indeterminate { 0%{width:0;margin-left:0} 50%{width:60%;margin-left:20%} 100%{width:0;margin-left:100%} }

        .console-wrap { margin-top:20px; }
        .console-header {
            display:flex; justify-content:space-between; align-items:center;
            background:#21262d; border:1px solid #30363d;
            border-bottom:none; border-radius:8px 8px 0 0;
            padding:8px 14px; font-size:0.82rem; color:#8b949e;
        }
        .console-dots { display:flex; gap:6px; }
        .console-dot { width:12px; height:12px; border-radius:50%; }
        .console-dot.r { background:#f85149; }
        .console-dot.y { background:#d29922; }
        .console-dot.g { background:#3fb950; }
        #consoleBox {
            background:#0d1117; color:#3fb950;
            font-family:'Courier New',Courier,monospace;
            padding:18px 20px; height:460px; overflow-y:auto;
            border:1px solid #30363d; border-radius:0 0 8px 8px;
            white-space:pre-wrap; font-size:13px; line-height:1.6;
        }
        #consoleBox .warn  { color:#d29922; }
        #consoleBox .error { color:#f85149; }
        #consoleBox .hi    { color:#58a6ff; }
        #consoleBox .good  { color:#3fb950; font-weight:bold; }

        #fidelityCard {
            display:none; margin-top:20px;
            background:#0d1d3a; border:1px solid #1f6feb;
            border-radius:8px; padding:18px 22px;
        }
        #fidelityCard h3 { margin:0 0 12px 0; color:#58a6ff; font-size:1rem; }
        .metric-row { display:flex; gap:30px; flex-wrap:wrap; }
        .metric { display:flex; flex-direction:column; }
        .metric .val { font-size:1.6rem; font-weight:700; color:#e6edf3; }
        .metric .lbl { font-size:0.78rem; color:#8b949e; text-transform:uppercase; }
        .metric .val.good { color:#3fb950; }
        .metric .val.ok   { color:#d29922; }
        .metric .val.bad  { color:#f85149; }
    </style>
</head>
<body>
<div class="container">
    <h1>🔬 Photonic Quantum Experiment Designer</h1>
    <p class="subtitle">AI-driven autonomous design of quantum optical circuits. Choose a model, pick a target state, and run.</p>

    <!-- ── MODEL SELECTOR ── -->
    <div class="model-banner" id="modelBanner">

        <div class="model-card selected" id="cardA" onclick="selectModel('A')">
            <div class="model-title">Model A — Fock Engine</div>
            <span class="model-tag">engine.py</span>
            <div class="model-desc">
                Full Fock-space simulation with 3 SPDC sources, 12 optical modes,
                sparse matrix expm. Supports Hong-Ou-Mandel interference and
                post-selection. Best for high-accuracy results.
            </div>
            <span class="model-badge">3 SPDC · Sparse · 8074-dim</span>
        </div>

        <div class="model-card" id="cardB" onclick="selectModel('B')">
            <div class="model-title">Model B — Alok Engine</div>
            <span class="model-tag">alok_final.py</span>
            <div class="model-desc">
                State-vector simulation with 2 SPDC sources, memory-safe
                gate application, Adam-optimised genetic search, and
                heralding / post-selection support.
            </div>
            <span class="model-badge">2 SPDC · State-vector · 4-qubit</span>
        </div>

    </div>

    <!-- ── CONTROLS ── -->
    <div class="controls">

        <!-- Target state — changes based on selected model -->
        <div class="control-group">
            <label>Target Quantum State</label>

            <!-- Model A targets -->
            <div id="targetA">
                <select id="targetSelectA" onchange="toggleCustomPanelA()">
                    <option value="bell">2-Qubit Bell State  |HH⟩+|VV⟩</option>
                    <option value="ghz4">4-Qubit GHZ State  |HHHH⟩+|VVVV⟩</option>
                    <option value="custom">Custom Amplitude Vector…</option>
                </select>
                <div id="customPanel">
                    <label>Amplitude Vector (8 values)</label>
                    <textarea id="customVecInput"
                        placeholder="Enter 8 comma-separated amplitudes:&#10;[0] HHH  [1] HHV  [2] HVH  [3] HVV&#10;[4] VHH  [5] VHV  [6] VVH  [7] VVV&#10;&#10;Example (GHZ): 1,0,0,0,0,0,0,1&#10;Example (W):   0,1,1,0,1,0,0,0"></textarea>
                    <div class="examples">⚠ Exactly 8 values. Auto-normalised. Complex OK: 1+0j</div>
                </div>
            </div>

            <!-- Model B targets -->
            <div id="targetB" style="display:none;">
                <select id="targetSelectB" onchange="toggleCustomPanelB()">
                    <option value="bell4"  data-vec="[1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]">2-Qubit Bell State</option>
                    <option value="ghz4"   data-vec="[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]">4-Qubit GHZ State</option>
                    <option value="w4"     data-vec="[0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0]">4-Qubit W State</option>
                    <option value="cluster" data-vec="[1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,-1]">4-Qubit Cluster State</option>
                    <option value="custom_b">Custom Target Vector…</option>
                </select>
                <div id="alokCustomPanel">
                    <label>Target State Vector</label>
                    <textarea id="alokCustomVecInput"
                        placeholder="Enter comma-separated amplitudes.&#10;4 values  → 2-qubit state&#10;8 values  → 3-qubit state&#10;16 values → 4-qubit state&#10;&#10;Example (Bell): 1,0,0,1&#10;Example (GHZ4): 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"></textarea>
                    <div class="examples">Auto-normalised. Complex OK.</div>
                </div>

                <!-- Custom input state toggle -->
                <div style="margin-top:12px; display:flex; align-items:center; gap:10px;">
                    <label style="text-transform:none; font-size:0.85rem; color:#8b949e; cursor:pointer;">
                        <input type="checkbox" id="customInputToggle" onchange="toggleAlokInputPanel()" style="margin-right:6px;">
                        Provide custom input state (default: SPDC)
                    </label>
                </div>
                <div id="alokInputPanel">
                    <label>Input State Vector</label>
                    <textarea id="alokInputVecInput"
                        placeholder="Default: SPDC Bell state(s). Override here.&#10;4 values  → 2-qubit&#10;16 values → 4-qubit&#10;&#10;Example: 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"></textarea>
                    <div class="examples">Leave blank to use SPDC default.</div>
                </div>
            </div>
        </div>

        <!-- GA parameters -->
        <div class="control-group">
            <label>GA Parameters</label>
            <div class="param-row">
                <div>
                    <input type="number" id="popInput" value="40" min="5" max="500">
                    <div class="hint">Population</div>
                </div>
                <div>
                    <input type="number" id="gensInput" value="100" min="1" max="10000">
                    <div class="hint">Generations</div>
                </div>
            </div>
            <div class="hint" style="margin-top:6px;">
                Pop=40 / Gens=100 → fast test.  Pop=120 / Gens=3000 → full run.
            </div>
        </div>

    </div>

    <!-- Action row -->
    <div class="btn-row">
        <button id="runBtn" onclick="startSim()">▶ Run Optimization</button>
        <button id="clearBtn" onclick="clearConsole()">⌫ Clear</button>
        <button id="downloadPdfBtn" onclick="downloadPdf()" disabled title="Run an optimization first">📄 PDF Report</button>
        <button id="downloadJsonBtn" onclick="downloadJson()" disabled title="Run an optimization first">⬇ JSON Results</button>
        <button id="downloadLogBtn" onclick="downloadLog()" disabled title="Run an optimization first">📋 Save Log</button>
        <span id="statusBadge">
            <span class="dot"></span>
            <span id="statusText">Ready</span>
        </span>
    </div>

    <div class="progress-wrap"><div id="progressBar"></div></div>

    <!-- Console -->
    <div class="console-wrap">
        <div class="console-header">
            <div class="console-dots">
                <div class="console-dot r"></div>
                <div class="console-dot y"></div>
                <div class="console-dot g"></div>
            </div>
            <span id="consoleModelLabel">simulation output</span>
            <span id="elapsedLabel"></span>
        </div>
        <div id="consoleBox">System ready. Select a model and target, then click ▶ Run Optimization.</div>
    </div>

    <!-- Fidelity card -->
    <div id="fidelityCard">
        <h3>✅ Optimization Complete</h3>
        <div class="metric-row">
            <div class="metric">
                <span class="val" id="mFidelity">—</span>
                <span class="lbl">Fidelity</span>
            </div>
            <div class="metric">
                <span class="val" id="mProb">—</span>
                <span class="lbl">Success Probability</span>
            </div>
            <div class="metric">
                <span class="val" id="mGates">—</span>
                <span class="lbl">Circuit Depth</span>
            </div>
            <div class="metric">
                <span class="val" id="mGrade">—</span>
                <span class="lbl">Grade</span>
            </div>
        </div>
    </div>
</div>

<script>
// ── State ─────────────────────────────────────────────────────────────────────
let selectedModel = 'A';

// ── Model selection ───────────────────────────────────────────────────────────
function selectModel(m) {
    selectedModel = m;
    document.getElementById('cardA').className = 'model-card' + (m==='A' ? ' selected' : '');
    document.getElementById('cardB').className = 'model-card' + (m==='B' ? ' selected' : '');
    document.getElementById('targetA').style.display = m==='A' ? 'block' : 'none';
    document.getElementById('targetB').style.display = m==='B' ? 'block' : 'none';
    document.getElementById('consoleModelLabel').textContent =
        m==='A' ? 'engine.py output' : 'alok_final.py output';
}

// ── Custom panel toggles ──────────────────────────────────────────────────────
function toggleCustomPanelA() {
    const val = document.getElementById('targetSelectA').value;
    document.getElementById('customPanel').className = val==='custom' ? 'visible' : '';
}

function toggleCustomPanelB() {
    const val = document.getElementById('targetSelectB').value;
    document.getElementById('alokCustomPanel').className = val==='custom_b' ? 'visible' : '';
}

function toggleAlokInputPanel() {
    const checked = document.getElementById('customInputToggle').checked;
    document.getElementById('alokInputPanel').className = checked ? 'visible' : '';
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setStatus(state, text) {
    const badge = document.getElementById('statusBadge');
    const dot   = badge.querySelector('.dot');
    document.getElementById('statusText').textContent = text;
    const styles = {
        running: { bg:'#0d2d0e', col:'#3fb950', brd:'#238636', dotCls:'dot pulse' },
        done:    { bg:'#0d1d3a', col:'#58a6ff', brd:'#1f6feb', dotCls:'dot' },
        error:   { bg:'#2d0d0d', col:'#f85149', brd:'#b91c1c', dotCls:'dot' },
        ready:   { bg:'#21262d', col:'#8b949e', brd:'#30363d', dotCls:'dot' },
    };
    const s = styles[state] || styles.ready;
    badge.style.background = s.bg;
    badge.style.color = s.col;
    badge.style.borderColor = s.brd;
    dot.className = s.dotCls;
}

function setProgress(indeterminate, pct) {
    const bar = document.getElementById('progressBar');
    if (indeterminate) { bar.style.width=''; bar.classList.add('indeterminate'); }
    else { bar.classList.remove('indeterminate'); bar.style.width = pct+'%'; }
}

function colorLine(raw) {
    if (!raw) return '';
    const e = raw.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    if (e.includes('✓')||e.includes('EXCELLENT')||e.includes('TARGET REACHED')||e.includes('PERFECT'))
        return `<span class="good">${e}</span>`;
    if (e.includes('[!]')||e.includes('Stuck')||e.includes('injecting'))
        return `<span class="warn">${e}</span>`;
    if (e.includes('Error')||e.includes('ERROR')||e.includes('failed'))
        return `<span class="error">${e}</span>`;
    if (e.includes('Gen ')||e.includes('Generation')||e.includes('EVOLUTION')||e.includes('Fidelity'))
        return `<span class="hi">${e}</span>`;
    return e;
}

function appendToConsole(text) {
    const box = document.getElementById('consoleBox');
    text.split('\n').forEach(line => {
        if (line.trim()) {
            box.innerHTML += colorLine(line) + '\n';
            _fullLog += line + '\n';
        }
    });
    box.scrollTop = box.scrollHeight;
}

function clearConsole() {
    document.getElementById('consoleBox').innerHTML =
        'System ready. Select a model and target, then click ▶ Run Optimization.\n';
    document.getElementById('fidelityCard').style.display = 'none';
    document.getElementById('downloadPdfBtn').disabled = true;
                document.getElementById('downloadJsonBtn').disabled = true;
                document.getElementById('downloadLogBtn').disabled = true;
    setProgress(false, 0);
    setStatus('ready', 'Ready');
    document.getElementById('elapsedLabel').textContent = '';
}

// ── Download functions ───────────────────────────────────────────────────────

// Store last result and full log for downloads
let _lastResult = null;
let _fullLog    = '';

function downloadPdf() {
    // Open PDF in new tab — Flask serves it from /download_pdf
    window.open('/download_pdf', '_blank');
}

function downloadJson() {
    if (!_lastResult) { alert('No result yet. Run an optimization first.'); return; }
    const blob = new Blob([JSON.stringify(_lastResult, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'quantum_experiment_result.json';
    a.click();
    URL.revokeObjectURL(url);
}

function downloadLog() {
    if (!_fullLog) { alert('No log yet. Run an optimization first.'); return; }
    const blob = new Blob([_fullLog], { type: 'text/plain' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'quantum_experiment_log.txt';
    a.click();
    URL.revokeObjectURL(url);
}

// ── Main run ──────────────────────────────────────────────────────────────────
function startSim() {
    const pop  = parseInt(document.getElementById('popInput').value)  || 40;
    const gens = parseInt(document.getElementById('gensInput').value) || 100;

    let url;

    if (selectedModel === 'A') {
        // ── Model A (engine.py) ──
        let target = document.getElementById('targetSelectA').value;
        let customVecStr = '';
        if (target === 'custom') {
            customVecStr = document.getElementById('customVecInput').value.trim();
            if (!customVecStr) { alert('Please enter a custom amplitude vector.'); return; }
            const parts = customVecStr.split(',').map(s=>s.trim()).filter(Boolean);
            if (parts.length !== 8) {
                alert(`Expected 8 values, got ${parts.length}.`); return;
            }
        }
        url = `/stream?model=A&target=${encodeURIComponent(target)}&pop=${pop}&gens=${gens}`;
        if (target === 'custom') url += `&vec=${encodeURIComponent(customVecStr)}`;

    } else {
        // ── Model B (alok_final.py) ──
        let targetVal = document.getElementById('targetSelectB').value;
        let targetVec = '';
        let inputVec  = '';

        if (targetVal === 'custom_b') {
            targetVec = document.getElementById('alokCustomVecInput').value.trim();
            if (!targetVec) { alert('Please enter a custom target vector.'); return; }
        } else {
            // Get the data-vec from the selected option
            const opt = document.querySelector(`#targetSelectB option[value="${targetVal}"]`);
            targetVec = opt ? opt.getAttribute('data-vec') : '';
        }

        if (document.getElementById('customInputToggle').checked) {
            inputVec = document.getElementById('alokInputVecInput').value.trim();
        }

        url = `/stream?model=B&target_vec=${encodeURIComponent(targetVec)}&pop=${pop}&gens=${gens}`;
        if (inputVec) url += `&input_vec=${encodeURIComponent(inputVec)}`;
    }

    // UI setup
    document.getElementById('consoleBox').innerHTML = '';
    _fullLog = '';
    document.getElementById('fidelityCard').style.display = 'none';
    document.getElementById('downloadPdfBtn').disabled = true;
            document.getElementById('downloadJsonBtn').disabled = true;
            document.getElementById('downloadLogBtn').disabled = true;
    document.getElementById('runBtn').disabled = true;
    setStatus('running', `Running Model ${selectedModel}…`);
    setProgress(true);
    appendToConsole(`▶ Starting Model ${selectedModel === 'A' ? 'A (engine.py)' : 'B (alok_final.py)'}...\n`);

    const startTime = Date.now();
    const timer = setInterval(() => {
        const s = ((Date.now()-startTime)/1000).toFixed(0);
        document.getElementById('elapsedLabel').textContent = `${s}s elapsed`;
    }, 1000);

    const evtSource = new EventSource(url);

    evtSource.onmessage = function(e) {
        let data;
        try { data = JSON.parse(e.data); } catch { appendToConsole(e.data); return; }
        if (data.log !== undefined) appendToConsole(data.log);
        if (data.done) {
            evtSource.close();
            clearInterval(timer);
            document.getElementById('runBtn').disabled = false;
            setProgress(false, 100);
            if (data.error) {
                setStatus('error', 'Error');
                appendToConsole('\n[Error] ' + data.error);
            } else {
                setStatus('done', 'Complete');
                appendToConsole('\n\n>> OPTIMIZATION COMPLETE.');
                document.getElementById('downloadPdfBtn').disabled = false;
                    document.getElementById('downloadJsonBtn').disabled = false;
                    document.getElementById('downloadLogBtn').disabled = false;
                if (data.result) {
                    const r = data.result;
                    _lastResult = r;
                    const fid = parseFloat(r.fidelity);
                    const fidEl = document.getElementById('mFidelity');
                    fidEl.textContent = fid.toFixed(4);
                    fidEl.className = 'val '+(fid>0.99?'good':fid>0.80?'ok':'bad');
                    document.getElementById('mProb').textContent  = parseFloat(r.probability||0).toFixed(6);
                    document.getElementById('mGates').textContent = r.gates;
                    const gradeEl = document.getElementById('mGrade');
                    gradeEl.textContent = r.grade;
                    gradeEl.className = 'val '+(fid>0.99?'good':fid>0.80?'ok':'bad');
                    document.getElementById('fidelityCard').style.display = 'block';
                }
            }
        }
    };

    evtSource.onerror = function() {
        evtSource.close();
        clearInterval(timer);
        appendToConsole('\n[Error] Connection to server lost.');
        document.getElementById('runBtn').disabled = false;
        setStatus('error', 'Connection lost');
        setProgress(false, 0);
    };
}
</script>
</body>
</html>"""

# ── Alok engine wrapper ────────────────────────────────────────────────────────

def run_alok_experiment(target_vec_str, input_vec_str, pop, gens, log_q, result_q):
    """
    Run alok_final.py genetic search and stream logs via log_q.
    target_vec_str : comma-separated amplitude string e.g. "[1,0,0,0,...,1]"
    input_vec_str  : comma-separated input state (empty = use SPDC default)
    """
    import ast
    import numpy as np

    def log(msg):
        log_q.put(str(msg))

    try:
        # Parse target vector
        target_list = ast.literal_eval(target_vec_str.strip())
        target = _alok.normalize(np.array(target_list, dtype=complex))

        # Determine n_qubits and default input state
        dim = len(target_list)
        n_qubits = int(np.log2(dim))

        if input_vec_str:
            input_list  = ast.literal_eval(input_vec_str.strip())
            input_state = _alok.normalize(np.array(input_list, dtype=complex))
            log(f"Using custom input state (dim={len(input_list)})")
        else:
            if dim == 4:
                input_state = _alok.spdc_single()
                log("Input: SPDC single source (Bell state)")
            elif dim == 16:
                input_state = _alok.spdc_two_sources()
                log("Input: 2x SPDC sources (Bell × Bell)")
            else:
                input_state = _alok.vacuum_state(n_qubits)
                log(f"Input: vacuum state ({n_qubits} qubits)")

        log(f"Target dim: {dim}  |  Qubits: {n_qubits}")
        log(f"Target: {np.round(target, 3)}")
        log("-" * 50)

        circuit, F = _alok.genetic_search(
            target         = target,
            input_state    = input_state,
            population     = pop,
            generations    = gens,
            circuit_length = 12,
            log_queue      = log_q,
            web_mode       = True,
        )

        # Print final circuit
        log("\nBest circuit found:")
        log("-" * 45)
        for i, (gate, param, qa, qb) in enumerate(circuit):
            if param is not None:
                log(f"  Step {i+1:2d}: {gate:6s} qubit {qa}  angle={param:.3f}r ({np.degrees(param):.1f}°)")
            else:
                log(f"  Step {i+1:2d}: {gate:6s} qubits {qa} → {qb}")
        log("-" * 45)
        log(f"\nFinal Fidelity: {F:.6f}")
        if   F > 0.999: log("✓ PERFECT — experiment found!")
        elif F > 0.99:  log("✓ EXCELLENT — nearly perfect!")
        elif F > 0.90:  log("GOOD — try more generations")
        else:           log("NEEDS IMPROVEMENT")

        n_spdc = 1 if dim <= 4 else 2
        result_q.put({
            'fidelity':     F,
            'probability':  1.0,
            'circuit':      circuit,
            'target_label': f'{dim}-dim state ({n_qubits} qubits)',
            'target_dim':   dim,
            'n_spdc':       n_spdc,
            'report_lines': [
                f"Model: alok_final.py",
                f"Fidelity: {F:.6f}",
                f"Circuit depth: {len(circuit)}",
                f"SPDC sources: {n_spdc}",
                f"Qubits: {n_qubits}",
            ],
        })

    except Exception as exc:
        import traceback
        log_q.put(f'[Exception] {exc}\n{traceback.format_exc()}')

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(_HTML)


@app.route('/stream')
def stream():
    model = request.args.get('model', 'A').strip().upper()

    try:
        pop  = max(5,  min(500,   int(request.args.get('pop',  40))))
        gens = max(1,  min(10000, int(request.args.get('gens', 100))))
    except ValueError:
        pop, gens = 40, 100

    # ── Model A ───────────────────────────────────────────────────────────────
    if model == 'A':
        if not ENGINE_A_OK:
            def err():
                yield f"data: {json.dumps({'log': f'[Error] engine.py failed: {_ENGINE_A_ERR}'})}\n\n"
                yield f"data: {json.dumps({'done': True, 'error': _ENGINE_A_ERR})}\n\n"
            return Response(err(), mimetype='text/event-stream')

        target_param = request.args.get('target', 'ghz4').strip()
        vec_param    = request.args.get('vec', '').strip()

        if target_param in ('ghz4', 'bell'):
            target_name, custom_vec = target_param, None
        elif target_param == 'custom':
            target_name = 'custom'
            custom_vec  = vec_param or None
        else:
            target_name, custom_vec = 'custom', target_param

        if custom_vec:
            try:
                parse_custom_vec(custom_vec)
            except ValueError as ve:
                msg = str(ve)
                def verr(m=msg):
                    yield f"data: {json.dumps({'log': '[ValueError] '+m})}\n\n"
                    yield f"data: {json.dumps({'done': True, 'error': m})}\n\n"
                return Response(verr(), mimetype='text/event-stream')

        log_q, result_q = queue.Queue(), queue.Queue()

        def worker_a():
            try:
                run_engine(
                    target_name=target_name, custom_vec=custom_vec,
                    pop_size=pop, gens=gens,
                    log_queue=log_q, result_queue=result_q,
                )
            except Exception as exc:
                import traceback
                log_q.put(f'[Exception] {exc}\n{traceback.format_exc()}')
            finally:
                log_q.put(None)

        threading.Thread(target=worker_a, daemon=True).start()

    # ── Model B ───────────────────────────────────────────────────────────────
    elif model == 'B':
        if not ENGINE_B_OK:
            def err():
                yield f"data: {json.dumps({'log': f'[Error] alok_final.py failed: {_ENGINE_B_ERR}'})}\n\n"
                yield f"data: {json.dumps({'done': True, 'error': _ENGINE_B_ERR})}\n\n"
            return Response(err(), mimetype='text/event-stream')

        target_vec = request.args.get('target_vec', '').strip()
        input_vec  = request.args.get('input_vec',  '').strip()

        if not target_vec:
            def no_vec():
                yield f"data: {json.dumps({'log': '[Error] No target vector provided.'})}\n\n"
                yield f"data: {json.dumps({'done': True, 'error': 'No target vector'})}\n\n"
            return Response(no_vec(), mimetype='text/event-stream')

        log_q, result_q = queue.Queue(), queue.Queue()

        def worker_b():
            try:
                run_alok_experiment(target_vec, input_vec, pop, gens, log_q, result_q)
            except Exception as exc:
                import traceback
                log_q.put(f'[Exception] {exc}\n{traceback.format_exc()}')
            finally:
                log_q.put(None)

        threading.Thread(target=worker_b, daemon=True).start()

    else:
        def bad():
            yield f"data: {json.dumps({'done': True, 'error': f'Unknown model: {model}'})}\n\n"
        return Response(bad(), mimetype='text/event-stream')

    # ── Shared SSE generator ──────────────────────────────────────────────────
    def generate():
        global _last_result
        # Send an immediate ping so browser knows connection is alive
        yield f"data: {json.dumps({'log': '>>> Model B (alok_final.py) connected'})}\n\n"
        while True:
            try:
                msg = log_q.get(timeout=3600)
            except queue.Empty:
                yield f"data: {json.dumps({'log': '[timeout]'})}\n\n"
                break

            if msg is None:   # sentinel
                result_data = None
                try:
                    res = result_q.get_nowait()
                    _last_result = res
                    result_data  = {
                        'fidelity':    res.get('fidelity',    0.0),
                        'probability': res.get('probability', 0.0),
                        'gates':       len(res.get('circuit', [])),
                        'grade':       _grade(res.get('fidelity', 0.0)),
                    }
                except queue.Empty:
                    pass

                payload = {'done': True}
                if result_data:
                    payload['result'] = result_data
                yield f"data: {json.dumps(payload)}\n\n"
                break

            for line in str(msg).splitlines(keepends=True):
                if line.strip():
                    yield f"data: {json.dumps({'log': line})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no', 'X-Content-Type-Options': 'nosniff'})


def _grade(f):
    if   f > 0.99: return 'EXCELLENT'
    elif f > 0.90: return 'GOOD'
    elif f > 0.70: return 'FAIR'
    else:          return 'NEEDS MORE EXPLORATION'


@app.route('/last_result')
def last_result():
    if not _last_result:
        return jsonify({'error': 'No result yet'}), 404
    safe = {k: v for k, v in _last_result.items() if k != 'circuit'}
    safe['circuit_depth'] = len(_last_result.get('circuit', []))
    return jsonify(safe)


@app.route('/download_pdf')
def download_pdf():
    if not _last_result:
        return 'No result yet. Run an optimization first.', 404
    try:
        return send_file(
            BytesIO(_build_pdf(_last_result)),
            mimetype='application/pdf',
            as_attachment=True,
            download_name='photonic_experiment_report.pdf',
        )
    except Exception as e:
        return f'PDF generation failed: {e}', 500


def _build_pdf(result):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles    import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units     import cm
    from reportlab.lib            import colors
    from reportlab.platypus       import (SimpleDocTemplate, Paragraph, Spacer,
                                          Table, TableStyle, HRFlowable)

    buf = BytesIO()
    margin = 2.2 * cm
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=margin, rightMargin=margin,
                            topMargin=margin,  bottomMargin=margin)
    styles = getSampleStyleSheet()
    title_s = ParagraphStyle('t', parent=styles['Title'],
                              fontSize=18, textColor=colors.HexColor('#1a1a2e'), spaceAfter=4)
    sub_s   = ParagraphStyle('s', parent=styles['Normal'],
                              fontSize=10, textColor=colors.HexColor('#555'), spaceAfter=12)
    head_s  = ParagraphStyle('h', parent=styles['Heading2'],
                              fontSize=12, textColor=colors.HexColor('#1a1a2e'),
                              spaceBefore=14, spaceAfter=6)
    body_s  = ParagraphStyle('b', parent=styles['Normal'],
                              fontSize=9, leading=14, fontName='Courier')

    fidelity    = result.get('fidelity',    0.0)
    probability = result.get('probability', 0.0)
    target_lbl  = result.get('target_label', '?')
    circuit     = result.get('circuit', [])
    report_lines = result.get('report_lines', [])

    story = []
    story.append(Paragraph('🔬 Photonic Quantum Experiment Report', title_s))
    story.append(Paragraph(f'Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}  |  Target: {target_lbl}', sub_s))
    story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#ddd')))
    story.append(Spacer(1, 10))
    story.append(Paragraph('Key Metrics', head_s))

    grade_bg  = (colors.HexColor('#d4edda') if fidelity > 0.99 else
                 colors.HexColor('#fff3cd') if fidelity > 0.80 else
                 colors.HexColor('#f8d7da'))
    metrics = [
        ['Metric', 'Value'],
        ['Fidelity',            f'{fidelity:.6f}'],
        ['Success Probability', f'{probability:.6f}'],
        ['Circuit Depth',       f'{len(circuit)} gates'],
        ['Grade',               _grade(fidelity)],
    ]
    t = Table(metrics, colWidths=[5*cm, 8*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('BACKGROUND', (0,4), (-1,4), grade_bg),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
        ('ROWPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))
    story.append(Paragraph('Discovered Circuit', head_s))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#ccc')))
    story.append(Spacer(1, 6))

    txt = '\n'.join(str(l) for l in report_lines) if report_lines else '(no report lines)'
    for line in txt.split('\n'):
        line = line.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        story.append(Paragraph(line or '&nbsp;', body_s))

    # ── Full circuit table ──────────────────────────────────────────────────
    if circuit:
        story.append(Spacer(1, 16))
        story.append(Paragraph('Circuit Gate Sequence', head_s))
        story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#ccc')))
        story.append(Spacer(1, 6))
        circ_data = [['Step', 'Gate', 'Qubit(s)', 'Parameter']]
        for i, gate_entry in enumerate(circuit):
            try:
                if len(gate_entry) == 4:
                    g, param, qa, qb = gate_entry
                    qubit_str = f"{qa} → {qb}" if qb is not None else str(qa)
                    param_str = f"{float(param):.4f} rad" if param is not None else "—"
                else:
                    g = str(gate_entry)
                    qubit_str, param_str = "—", "—"
            except:
                g = str(gate_entry); qubit_str = "—"; param_str = "—"
            circ_data.append([str(i+1), str(g), qubit_str, param_str])

        ct = Table(circ_data, colWidths=[1.2*cm, 3*cm, 4*cm, 4.5*cm])
        ct.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
            ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0,0), (-1,-1), 8),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('GRID',          (0,0), (-1,-1), 0.4, colors.HexColor('#dee2e6')),
            ('ROWPADDING',    (0,0), (-1,-1), 5),
            ('FONTNAME',      (0,1), (0,-1), 'Helvetica-Bold'),
        ]))
        story.append(ct)

    # ── Optical Tools Used ─────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(Paragraph('Optical Components Used', head_s))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#ccc')))
    story.append(Spacer(1, 6))

    # Gate descriptions lookup
    GATE_INFO = {
        'HWP':   ('Half-Wave Plate',        'Single qubit', 'Rotates polarization by 2theta. At theta=22.5 acts as Hadamard. At theta=45 flips H to V.'),
        'QWP':   ('Quarter-Wave Plate',     'Single qubit', 'Introduces 90 degree phase shift. Converts linear to circular polarization and vice versa.'),
        'PHASE': ('Phase Shifter',          'Single qubit', 'Applies phase shift phi to the vertical component. Controls relative phase between H and V.'),
        'CNOT':  ('Controlled-NOT Gate',    'Two qubit',    'Flips target qubit when control is |1>. Primary gate for creating entanglement.'),
        'CZ':    ('Controlled-Z Gate',      'Two qubit',    'Applies -1 phase when both qubits are |1>. Used for cluster and graph state preparation.'),
        'SWAP':  ('SWAP Gate',              'Two qubit',    'Exchanges two qubits. Used to route photons between different spatial modes.'),
        'BS':    ('50:50 Beam Splitter',    'Two qubit',    'Coherently mixes two spatial modes. Creates Hong-Ou-Mandel interference between photons.'),
        'PS':    ('Phase Shifter',          'Single mode',  'Applies a phase to one optical mode. Controls interference in the Fock-space circuit.'),
        'DETECT':('Threshold Detector',     'Measurement',  'Post-selection step. Heralds a photon in a mode. Conditions the output state on a click.'),
    }

    # Count how many times each gate is used
    gate_counts = {}
    for gate_entry in circuit:
        try:
            if isinstance(gate_entry, dict):
                g = gate_entry.get('type', '?')
            elif len(gate_entry) >= 1:
                g = str(gate_entry[0])
            else:
                g = str(gate_entry)
            gate_counts[g] = gate_counts.get(g, 0) + 1
        except:
            pass

    if gate_counts:
        tools_data = [['Component', 'Type', 'Count', 'Physical Role']]
        for gate_name, count in sorted(gate_counts.items(), key=lambda x: -x[1]):
            info = GATE_INFO.get(gate_name, (gate_name, 'Unknown', 'No description available.'))
            tools_data.append([
                info[0],
                info[1],
                str(count),
                info[2],
            ])

        tools_tbl = Table(tools_data, colWidths=[3.8*cm, 2.2*cm, 1.2*cm, 6.5*cm])
        tools_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
            ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0,0), (-1,-1), 8),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, colors.HexColor('#f0f4ff')]),
            ('GRID',          (0,0), (-1,-1), 0.4, colors.HexColor('#dee2e6')),
            ('ROWPADDING',    (0,0), (-1,-1), 5),
            ('VALIGN',        (0,0), (-1,-1), 'TOP'),
            ('FONTNAME',      (0,1), (0,-1), 'Helvetica-Bold'),
            ('TEXTCOLOR',     (2,1), (2,-1), colors.HexColor('#1a1a2e')),
        ]))
        story.append(tools_tbl)
    else:
        story.append(Paragraph('No gate data available.', body_s))

    # ── SPDC Sources section ─────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(Paragraph('Photon Sources (SPDC)', head_s))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#ccc')))
    story.append(Spacer(1, 6))

    spdc_label = result.get('spdc_source', None)
    n_spdc     = result.get('n_spdc', None)

    # Infer from circuit size / target dim
    if n_spdc is None:
        circ_len = len(circuit)
        tgt_dim  = result.get('target_dim', 0)
        if tgt_dim == 4 or circ_len < 8:
            n_spdc = 1
        elif tgt_dim <= 16:
            n_spdc = 2
        else:
            n_spdc = 3

    spdc_rows = [['Source', 'Photons', 'Output State', 'Description']]
    for i in range(n_spdc):
        spdc_rows.append([
            f'SPDC {i+1}',
            '2',
            '(|HH> + |VV>) / sqrt(2)',
            'Bell state Phi+. Entangled photon pair from nonlinear crystal via spontaneous parametric down-conversion.',
        ])

    spdc_tbl = Table(spdc_rows, colWidths=[1.8*cm, 1.8*cm, 4.5*cm, 5.6*cm])
    spdc_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#0f6674')),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.HexColor('#e8f8f5'), colors.white]),
        ('GRID',          (0,0), (-1,-1), 0.4, colors.HexColor('#dee2e6')),
        ('ROWPADDING',    (0,0), (-1,-1), 5),
        ('VALIGN',        (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(spdc_tbl)

    # ── Component summary bar ─────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(Paragraph('Experiment Summary', head_s))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#ccc')))
    story.append(Spacer(1, 6))

    single_q = sum(v for k,v in gate_counts.items() if k in ('HWP','QWP','PHASE','PS'))
    two_q    = sum(v for k,v in gate_counts.items() if k in ('CNOT','CZ','SWAP','BS'))
    detectors= gate_counts.get('DETECT', 0)
    total    = len(circuit)

    summary_data = [
        ['Category',              'Count', 'Components'],
        ['Single-qubit gates',    str(single_q), 'HWP, QWP, Phase Shifter'],
        ['Two-qubit gates',       str(two_q),    'Beam Splitter, CNOT, CZ, SWAP'],
        ['Detectors / post-sel.', str(detectors),'Threshold Detector'],
        ['SPDC sources',          str(n_spdc),   'Bell-state photon pair generators'],
        ['Total circuit depth',   str(total),    'All gates combined'],
    ]
    sum_tbl = Table(summary_data, colWidths=[5*cm, 2*cm, 6.7*cm])
    sum_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('GRID',          (0,0), (-1,-1), 0.4, colors.HexColor('#dee2e6')),
        ('ROWPADDING',    (0,0), (-1,-1), 5),
        ('FONTNAME',      (0,1), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',      (1,1), (1,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',     (1,1), (1,-1), colors.HexColor('#1a1a2e')),
    ]))
    story.append(sum_tbl)

    doc.build(story)
    return buf.getvalue()


if __name__ == '__main__':
    print("Starting Photonic Quantum Experiment Designer...")
    print(f"  Model A (engine.py):    {'OK' if ENGINE_A_OK else 'MISSING'}")
    print(f"  Model B (alok_final.py): {'OK' if ENGINE_B_OK else 'MISSING'}")
    print("Open  http://localhost:5000  in your browser.")
    app.run(debug=False, host='0.0.0.0', port=5005, threaded=True)