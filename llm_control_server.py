"""
LLM Model Launcher Control Server v2
- API key authentication
- GPU/system monitoring
- Health checks for llama-server
- Log tailing
- Local network access
"""

VERSION = "2.1.0"

import subprocess
import sys
import os
import json
import time
import threading
import re
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import socket

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def validate_config(config):
    """Validate configuration and check that required paths exist"""
    errors = []
    warnings = []

    # Check required top-level keys
    required_keys = ["server", "security", "paths", "models"]
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required config key: '{key}'")

    if errors:
        return errors, warnings

    # Check server config
    if "host" not in config["server"] or "port" not in config["server"]:
        errors.append("Missing 'host' or 'port' in server config")
    if "llama_host" not in config["server"] or "llama_port" not in config["server"]:
        errors.append("Missing 'llama_host' or 'llama_port' in server config")

    # Check security config
    if "api_key" not in config["security"]:
        errors.append("Missing 'api_key' in security config")
    elif config["security"]["api_key"] == "CHANGE_ME_TO_SOMETHING_RANDOM_1234567890":
        warnings.append("Using default API key - please change for security!")

    # Check paths exist
    if "llama_server" in config["paths"]:
        llama_path = config["paths"]["llama_server"]
        if not os.path.exists(llama_path):
            errors.append(f"llama-server not found at: {llama_path}")
    else:
        errors.append("Missing 'llama_server' path in config")

    if "models_base" in config["paths"]:
        models_base = config["paths"]["models_base"]
        if not os.path.exists(models_base):
            errors.append(f"Models directory not found at: {models_base}")
    else:
        errors.append("Missing 'models_base' path in config")

    # Check models
    if not config["models"]:
        warnings.append("No models defined in config")
    else:
        for model_id, model_cfg in config["models"].items():
            if "name" not in model_cfg:
                warnings.append(f"Model '{model_id}' missing 'name' field")
            if "file" not in model_cfg:
                errors.append(f"Model '{model_id}' missing 'file' field")
            elif "models_base" in config["paths"]:
                model_path = os.path.join(
                    config["paths"]["models_base"], model_cfg["file"]
                )
                if not os.path.exists(model_path):
                    warnings.append(f"Model file not found: {model_path}")

    return errors, warnings


CONFIG = load_config()
config_errors, config_warnings = validate_config(CONFIG)

# Global state
current_process = None
current_model = None
process_logs = deque(maxlen=200)  # Keep last 200 log lines
start_time = None
request_count = 0


def get_model_path(model_cfg):
    """Build full path from config"""
    return os.path.join(CONFIG["paths"]["models_base"], model_cfg["file"])


def build_llama_args(model_id):
    """Build llama-server command line args from config"""
    model = CONFIG["models"][model_id]
    args = [
        CONFIG["paths"]["llama_server"],
        "-m",
        get_model_path(model),
        "--host",
        CONFIG["server"]["llama_host"],
        "--port",
        str(CONFIG["server"]["llama_port"]),
        "-c",
        str(model.get("context", 8192)),
        "-ngl",
        str(model.get("gpu_layers", 48)),
    ]

    if "cpu_moe" in model:
        args.extend(["--n-cpu-moe", str(model["cpu_moe"])])
    if "temp" in model:
        args.extend(["--temp", str(model["temp"])])
    if "top_k" in model:
        args.extend(["--top-k", str(model["top_k"])])
    if "top_p" in model:
        args.extend(["--top-p", str(model["top_p"])])
    if "min_p" in model:
        args.extend(["--min-p", str(model["min_p"])])
    if "extra_args" in model:
        args.extend(model["extra_args"])

    return args


def log_output_reader(pipe, log_type):
    """Read process output and store in log buffer"""
    try:
        for line in iter(pipe.readline, b""):
            try:
                decoded = line.decode("utf-8", errors="replace").strip()
                if decoded:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    process_logs.append(f"[{timestamp}] {decoded}")
            except Exception:
                pass
    except Exception:
        pass


def start_model(model_id):
    global current_process, current_model, process_logs, start_time

    if model_id not in CONFIG["models"]:
        return {"success": False, "error": f"Unknown model: {model_id}"}

    # Stop any running model first
    if current_process:
        stop_model()

    process_logs.clear()
    model = CONFIG["models"][model_id]
    cmd = build_llama_args(model_id)

    process_logs.append(
        f"[{datetime.now().strftime('%H:%M:%S')}] Starting {model['name']}..."
    )
    process_logs.append(
        f"[{datetime.now().strftime('%H:%M:%S')}] Command: {' '.join(cmd)}"
    )

    try:
        current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            if sys.platform == "win32"
            else 0,
        )
        current_model = model_id
        start_time = time.time()

        # Start log reader thread
        log_thread = threading.Thread(
            target=log_output_reader,
            args=(current_process.stdout, "stdout"),
            daemon=True,
        )
        log_thread.start()

        return {
            "success": True,
            "model": model_id,
            "name": model["name"],
            "pid": current_process.pid,
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": f"llama-server not found at {CONFIG['paths']['llama_server']}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def stop_model():
    global current_process, current_model, start_time

    if current_process is None:
        return {"success": True, "message": "No model running"}

    stopped_model = current_model
    stopped_model_name = (
        CONFIG["models"][current_model]["name"] if current_model else "Unknown"
    )
    process_logs.append(
        f"[{datetime.now().strftime('%H:%M:%S')}] Stopping {stopped_model_name}..."
    )
    try:
        if sys.platform == "win32":
            current_process.terminate()
        else:
            import signal

            current_process.send_signal(signal.SIGTERM)
        current_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        current_process.kill()
        current_process.wait()
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        current_process = None
        current_model = None
        start_time = None

    process_logs.append(
        f"[{datetime.now().strftime('%H:%M:%S')}] Stopped {stopped_model_name}"
    )
    return {"success": True, "stopped": stopped_model, "name": stopped_model_name}


def check_llama_health():
    """Check if llama-server is responding"""
    if current_process is None:
        return {"healthy": False, "reason": "not_running"}

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(
            (CONFIG["server"]["llama_host"], CONFIG["server"]["llama_port"])
        )
        sock.close()

        if result == 0:
            return {"healthy": True}
        else:
            return {"healthy": False, "reason": "port_not_responding"}
    except Exception as e:
        return {"healthy": False, "reason": str(e)}


def get_system_stats():
    """Get GPU and system stats (Windows)"""
    stats = {"gpu": None, "memory": None}

    try:
        # Try nvidia-smi for GPU stats
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 5:
                stats["gpu"] = {
                    "name": parts[0],
                    "vram_used_mb": int(parts[1]),
                    "vram_total_mb": int(parts[2]),
                    "utilization": int(parts[3]),
                    "temp_c": int(parts[4]),
                }
    except Exception:
        pass

    try:
        # System memory via wmic (Windows)
        result = subprocess.run(
            [
                "wmic",
                "OS",
                "get",
                "FreePhysicalMemory,TotalVisibleMemorySize",
                "/Value",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            free_match = re.search(r"FreePhysicalMemory=(\d+)", result.stdout)
            total_match = re.search(r"TotalVisibleMemorySize=(\d+)", result.stdout)
            if free_match and total_match:
                free_kb = int(free_match.group(1))
                total_kb = int(total_match.group(1))
                stats["memory"] = {
                    "used_gb": round((total_kb - free_kb) / 1024 / 1024, 1),
                    "total_gb": round(total_kb / 1024 / 1024, 1),
                }
    except Exception:
        pass

    return stats


def get_status():
    global request_count
    request_count += 1

    if current_process is None:
        return {"running": False, "request_count": request_count}

    # Check if process is still alive
    if current_process.poll() is not None:
        exit_code = current_process.returncode
        return {
            "running": False,
            "message": f"Process exited with code {exit_code}",
            "request_count": request_count,
        }

    uptime = int(time.time() - start_time) if start_time else 0
    health = check_llama_health()

    return {
        "running": True,
        "model": current_model,
        "name": CONFIG["models"][current_model]["name"],
        "pid": current_process.pid,
        "uptime_seconds": uptime,
        "health": health,
        "request_count": request_count,
    }


def get_logs(lines=50):
    """Get recent log lines"""
    log_list = list(process_logs)
    return log_list[-lines:] if len(log_list) > lines else log_list


def get_models():
    return {
        k: {"name": v["name"], "context": v.get("context", 8192)}
        for k, v in CONFIG["models"].items()
    }


def get_network_info():
    """Get network addresses for connection info"""
    info = {
        "local": None,
        "hostname": socket.gethostname(),
        "tailscale_ip": None,
        "tailscale_dns": None
    }

    try:
        # Get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        info["local"] = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    return info


# HTML UI
HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Launcher v2</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 1rem;
        }
        .container { max-width: 800px; margin: 0 auto; }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
            gap: 1rem;
        }
        h1 { font-weight: 400; font-size: 1.5rem; color: #fff; }
        .auth-status {
            font-size: 0.8rem;
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
            background: rgba(46, 213, 115, 0.2);
            color: #2ed573;
        }
        .auth-status.unauthorized {
            background: rgba(255, 71, 87, 0.2);
            color: #ff4757;
        }
        
        .grid { display: grid; gap: 1rem; grid-template-columns: 1fr; }
        @media (min-width: 600px) { .grid { grid-template-columns: 1fr 1fr; } }
        
        .card {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 1.25rem;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .card h2 {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #888;
            margin-bottom: 1rem;
        }
        
        .status-row {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.75rem;
        }
        .dot {
            width: 10px; height: 10px;
            border-radius: 50%;
            background: #ff4757;
            flex-shrink: 0;
        }
        .dot.running { background: #2ed573; animation: pulse 2s infinite; }
        .dot.warning { background: #ffa502; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        
        .model-name { color: #74b9ff; font-weight: 500; }
        .stat-value { font-size: 1.5rem; font-weight: 300; color: #fff; }
        .stat-label { font-size: 0.75rem; color: #888; }
        .stat-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; }
        
        .progress-bar {
            height: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            overflow: hidden;
            margin-top: 0.5rem;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #74b9ff, #a29bfe);
            transition: width 0.3s;
        }
        .progress-fill.high { background: linear-gradient(90deg, #ffa502, #ff4757); }
        
        .model-list { display: grid; gap: 0.75rem; }
        .model-btn {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 1rem;
            cursor: pointer;
            transition: all 0.2s;
            text-align: left;
            color: inherit;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .model-btn:hover {
            background: rgba(116, 185, 255, 0.1);
            border-color: #74b9ff;
        }
        .model-btn.active {
            background: rgba(46, 213, 115, 0.1);
            border-color: #2ed573;
        }
        .model-btn h3 { font-size: 0.95rem; font-weight: 500; color: #fff; }
        .model-btn .ctx { font-size: 0.75rem; color: #888; font-family: monospace; }
        
        .stop-btn {
            width: 100%;
            padding: 0.9rem;
            margin-top: 1rem;
            background: transparent;
            border: 1px solid #ff4757;
            border-radius: 8px;
            color: #ff4757;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .stop-btn:hover { background: rgba(255, 71, 87, 0.15); }
        .stop-btn:disabled { opacity: 0.3; cursor: not-allowed; }
        
        .logs-card { grid-column: 1 / -1; }
        .logs {
            background: #0a0a12;
            border-radius: 6px;
            padding: 0.75rem;
            font-family: "SF Mono", Monaco, monospace;
            font-size: 0.75rem;
            line-height: 1.5;
            max-height: 200px;
            overflow-y: auto;
            color: #888;
        }
        .logs .line { white-space: pre-wrap; word-break: break-all; }
        .logs .line.error { color: #ff4757; }
        .logs .line.success { color: #2ed573; }
        
        .network-info {
            font-size: 0.8rem;
            color: #666;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255,255,255,0.05);
        }
        .network-info code {
            background: rgba(255,255,255,0.1);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-size: 0.75rem;
        }
        
        .toast {
            position: fixed;
            bottom: 1.5rem;
            left: 50%;
            transform: translateX(-50%);
            background: #222;
            padding: 0.8rem 1.5rem;
            border-radius: 8px;
            font-size: 0.9rem;
            opacity: 0;
            transition: opacity 0.3s;
            z-index: 100;
        }
        .toast.show { opacity: 1; }
        
        .auth-modal {
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 200;
        }
        .auth-modal.hidden { display: none; }
        .auth-box {
            background: #1a1a2e;
            padding: 2rem;
            border-radius: 12px;
            width: 90%;
            max-width: 350px;
        }
        .auth-box h2 { margin-bottom: 1rem; font-weight: 400; }
        .auth-box input {
            width: 100%;
            padding: 0.8rem;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 6px;
            background: rgba(255,255,255,0.05);
            color: #fff;
            font-size: 1rem;
            margin-bottom: 1rem;
        }
        .auth-box button {
            width: 100%;
            padding: 0.8rem;
            background: #74b9ff;
            border: none;
            border-radius: 6px;
            color: #000;
            font-size: 1rem;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🖥️ LLM Launcher</h1>
            <div class="auth-status" id="authStatus">Authenticating...</div>
        </header>
        
        <div class="grid">
            <div class="card">
                <h2>Status</h2>
                <div class="status-row">
                    <div class="dot" id="statusDot"></div>
                    <span id="statusText">Checking...</span>
                </div>
                <div id="healthRow" class="status-row" style="display:none;">
                    <div class="dot" id="healthDot"></div>
                    <span id="healthText">Health: Unknown</span>
                </div>
                <div id="uptimeRow" style="display:none; margin-top:0.5rem; font-size:0.85rem; color:#888;">
                    Uptime: <span id="uptime">-</span>
                </div>
            </div>
            
            <div class="card">
                <h2>System</h2>
                <div class="stat-grid">
                    <div>
                        <div class="stat-value" id="gpuUtil">--%</div>
                        <div class="stat-label">GPU Usage</div>
                    </div>
                    <div>
                        <div class="stat-value" id="gpuTemp">--°C</div>
                        <div class="stat-label">GPU Temp</div>
                    </div>
                </div>
                <div style="margin-top:1rem;">
                    <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#888;">
                        <span>VRAM</span>
                        <span id="vramText">-- / -- GB</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="vramBar" style="width:0%"></div>
                    </div>
                </div>
            </div>
            
            <div class="card" style="grid-column: 1 / -1;">
                <h2>Models</h2>
                <div class="model-list" id="modelList"></div>
                <button class="stop-btn" id="stopBtn" onclick="stopModel()" disabled>
                    ⏹ Stop Current Model
                </button>
            </div>
            
            <div class="card logs-card">
                <h2>Logs</h2>
                <div class="logs" id="logs">Waiting for logs...</div>
            </div>
        </div>
        
        <div class="network-info" id="networkInfo"></div>
    </div>
    
    <div class="toast" id="toast"></div>
    
    <div class="auth-modal" id="authModal">
        <div class="auth-box">
            <h2>🔐 Enter API Key</h2>
            <input type="password" id="apiKeyInput" placeholder="API Key" autocomplete="off">
            <button onclick="authenticate()">Connect</button>
        </div>
    </div>

    <script>
        let apiKey = localStorage.getItem('llm_launcher_key') || '';
        let currentModel = null;

        async function api(endpoint, method = 'GET', timeoutMs = 10000) {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

            try {
                const res = await fetch(endpoint, {
                    method,
                    headers: { 'X-API-Key': apiKey },
                    signal: controller.signal
                });
                clearTimeout(timeoutId);

                if (res.status === 401) {
                    showAuthModal();
                    throw new Error('Unauthorized');
                }
                if (!res.ok) {
                    throw new Error(`HTTP ${res.status}`);
                }
                return res.json();
            } catch (e) {
                clearTimeout(timeoutId);
                if (e.name === 'AbortError') {
                    console.error(`Request timeout: ${endpoint}`);
                    throw new Error('Request timeout - check network connection');
                }
                throw e;
            }
        }
        
        function showAuthModal() {
            document.getElementById('authModal').classList.remove('hidden');
            document.getElementById('authStatus').textContent = 'Not authenticated';
            document.getElementById('authStatus').classList.add('unauthorized');
        }
        
        function hideAuthModal() {
            document.getElementById('authModal').classList.add('hidden');
            document.getElementById('authStatus').textContent = '✓ Authenticated';
            document.getElementById('authStatus').classList.remove('unauthorized');
        }
        
        async function authenticate() {
            const btn = event.target;
            const input = document.getElementById('apiKeyInput');
            apiKey = input.value;

            btn.textContent = 'Connecting...';
            btn.disabled = true;

            try {
                await api('/api/status');
                localStorage.setItem('llm_launcher_key', apiKey);
                hideAuthModal();
                init();
            } catch (e) {
                console.error('Authentication error:', e);
                showToast(e.message || 'Authentication failed - check API key and connection');
                btn.textContent = 'Connect';
                btn.disabled = false;
            }
        }
        
        function showToast(msg) {
            const toast = document.getElementById('toast');
            toast.textContent = msg;
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 3000);
        }
        
        function formatUptime(seconds) {
            if (seconds < 60) return `${seconds}s`;
            if (seconds < 3600) return `${Math.floor(seconds/60)}m ${seconds%60}s`;
            return `${Math.floor(seconds/3600)}h ${Math.floor((seconds%3600)/60)}m`;
        }
        
        function updateStatus(status) {
            const dot = document.getElementById('statusDot');
            const text = document.getElementById('statusText');
            const stopBtn = document.getElementById('stopBtn');
            const healthRow = document.getElementById('healthRow');
            const uptimeRow = document.getElementById('uptimeRow');
            
            if (status.running) {
                dot.classList.add('running');
                text.innerHTML = `Running: <span class="model-name">${status.name}</span>`;
                stopBtn.disabled = false;
                currentModel = status.model;
                
                // Health
                healthRow.style.display = 'flex';
                const healthDot = document.getElementById('healthDot');
                const healthText = document.getElementById('healthText');
                if (status.health?.healthy) {
                    healthDot.classList.add('running');
                    healthDot.classList.remove('warning');
                    healthText.textContent = 'Inference ready';
                } else {
                    healthDot.classList.remove('running');
                    healthDot.classList.add('warning');
                    healthText.textContent = 'Loading model...';
                }
                
                // Uptime
                uptimeRow.style.display = 'block';
                document.getElementById('uptime').textContent = formatUptime(status.uptime_seconds);
            } else {
                dot.classList.remove('running');
                text.textContent = status.message || 'No model running';
                stopBtn.disabled = true;
                currentModel = null;
                healthRow.style.display = 'none';
                uptimeRow.style.display = 'none';
            }
            
            document.querySelectorAll('.model-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.model === currentModel);
            });
        }
        
        function updateStats(stats) {
            if (stats.gpu) {
                document.getElementById('gpuUtil').textContent = `${stats.gpu.utilization}%`;
                document.getElementById('gpuTemp').textContent = `${stats.gpu.temp_c}°C`;
                const vramPct = Math.round((stats.gpu.vram_used_mb / stats.gpu.vram_total_mb) * 100);
                document.getElementById('vramText').textContent = 
                    `${(stats.gpu.vram_used_mb/1024).toFixed(1)} / ${(stats.gpu.vram_total_mb/1024).toFixed(1)} GB`;
                const vramBar = document.getElementById('vramBar');
                vramBar.style.width = `${vramPct}%`;
                vramBar.classList.toggle('high', vramPct > 85);
            }
        }
        
        function updateLogs(logs) {
            const container = document.getElementById('logs');
            container.innerHTML = logs.map(line => {
                let cls = 'line';
                if (line.includes('error') || line.includes('Error')) cls += ' error';
                if (line.includes('listening') || line.includes('ready')) cls += ' success';
                return `<div class="${cls}">${line}</div>`;
            }).join('');
            container.scrollTop = container.scrollHeight;
        }
        
        async function loadModels() {
            const models = await api('/api/models');
            const list = document.getElementById('modelList');
            list.innerHTML = '';
            
            for (const [id, info] of Object.entries(models)) {
                const btn = document.createElement('button');
                btn.className = 'model-btn';
                btn.dataset.model = id;
                btn.innerHTML = `
                    <div><h3>${info.name}</h3></div>
                    <span class="ctx">${(info.context/1024).toFixed(0)}K ctx</span>
                `;
                btn.onclick = () => startModel(id);
                list.appendChild(btn);
            }
        }
        
        async function loadNetworkInfo() {
            const info = await api('/api/network');
            const container = document.getElementById('networkInfo');
            let html = `<strong>${info.hostname}</strong>`;
            if (info.local) {
                html += ` · Local: <code>${info.local}:8081</code>`;
            }
            if (info.tailscale_dns) {
                html += ` · Tailscale: <code>${info.tailscale_dns}:8081</code>`;
            }
            container.innerHTML = html;
        }
        
        async function refresh() {
            try {
                const [status, stats, logs] = await Promise.all([
                    api('/api/status'),
                    api('/api/stats'),
                    api('/api/logs?lines=30')
                ]);
                updateStatus(status);
                updateStats(stats);
                updateLogs(logs);
            } catch (e) {
                if (e.message !== 'Unauthorized') {
                    console.error('Refresh error:', e);
                    // Don't show auth modal on network errors, just log them
                    if (e.message && e.message.includes('timeout')) {
                        console.warn('Connection timeout - will retry on next refresh');
                    }
                }
            }
        }
        
        async function startModel(modelId) {
            showToast(`Starting ${modelId}...`);
            try {
                const result = await api(`/api/start/${modelId}`, 'POST');
                if (result.success) {
                    showToast(`✓ ${result.name} starting`);
                } else {
                    showToast(`✗ ${result.error}`);
                }
            } catch (e) {}
            refresh();
        }
        
        async function stopModel() {
            showToast('Stopping model...');
            try {
                const result = await api('/api/stop', 'POST');
                showToast(result.success ? '✓ Stopped' : `✗ ${result.error}`);
            } catch (e) {}
            refresh();
        }
        
        async function init() {
            try {
                console.log('Initializing...');
                await api('/api/status');
                hideAuthModal();

                console.log('Loading models...');
                await loadModels();

                console.log('Loading network info...');
                await loadNetworkInfo();

                console.log('Initial refresh...');
                await refresh();

                console.log('Starting auto-refresh...');
                setInterval(refresh, 3000);

                console.log('Initialization complete');
            } catch (e) {
                console.error('Initialization failed:', e);
                if (e.message === 'Unauthorized') {
                    showAuthModal();
                } else {
                    showToast('Connection failed: ' + (e.message || 'Unknown error'));
                }
            }
        }
        
        // Check for saved key on load
        if (apiKey) {
            init();
        } else {
            showAuthModal();
        }
        
        // Allow enter key in auth input
        document.getElementById('apiKeyInput').addEventListener('keyup', e => {
            if (e.key === 'Enter') authenticate();
        });
    </script>
</body>
</html>"""


class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Log requests with client IP for debugging
        client_ip = self.client_address[0]
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {client_ip} - {format % args}")

    def check_auth(self):
        """Verify API key"""
        key = self.headers.get("X-API-Key", "")
        return key == CONFIG["security"]["api_key"]

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "X-API-Key")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def send_html(self, html):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # Public routes (no auth needed)
        if path in ["/", "/index.html"]:
            self.send_html(HTML_UI)
            return

        # Protected routes
        if not self.check_auth():
            self.send_json({"error": "Unauthorized"}, 401)
            return

        if path == "/api/status":
            self.send_json(get_status())
        elif path == "/api/models":
            self.send_json(get_models())
        elif path == "/api/stats":
            self.send_json(get_system_stats())
        elif path == "/api/logs":
            params = parse_qs(parsed.query)
            lines = int(params.get("lines", [50])[0])
            self.send_json(get_logs(lines))
        elif path == "/api/network":
            self.send_json(get_network_info())
        elif path == "/api/version":
            self.send_json({"version": VERSION, "name": "LLM Launcher Control Server"})
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if not self.check_auth():
            self.send_json({"error": "Unauthorized"}, 401)
            return

        if self.path == "/api/stop":
            self.send_json(stop_model())
        elif self.path.startswith("/api/start/"):
            model_id = self.path.split("/")[-1]
            self.send_json(start_model(model_id))
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "X-API-Key")
        self.end_headers()


def main():
    # Reload config in case it was edited
    global CONFIG, config_errors, config_warnings
    CONFIG = load_config()
    config_errors, config_warnings = validate_config(CONFIG)

    # Print validation results
    if config_errors:
        print("❌ Configuration Errors:")
        for error in config_errors:
            print(f"   - {error}")
        print("\nPlease fix the errors in config.json and try again.\n")
        sys.exit(1)

    if config_warnings:
        print("⚠️  Configuration Warnings:")
        for warning in config_warnings:
            print(f"   - {warning}")
        print()

    host = CONFIG["server"]["host"]
    port = CONFIG["server"]["port"]

    server = HTTPServer((host, port), RequestHandler)

    net_info = get_network_info()

    tailscale_url = f"http://{net_info.get('tailscale_dns', 'N/A')}:{port}"
    local_url = f"http://{net_info.get('local', 'localhost')}:{port}"

    print(f"""
==================================================================
                 LLM Launcher Control Server v2.1
==================================================================
  Local:      {local_url}
  Tailscale:  {tailscale_url}

  API Key:    {CONFIG["security"]["api_key"][:20]}...

  Press Ctrl+C to stop
==================================================================
""")

    if CONFIG["security"]["api_key"] == "CHANGE_ME_TO_SOMETHING_RANDOM_1234567890":
        print(
            "⚠️  WARNING: Using default API key! Edit config.json to set a secure key.\n"
        )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        result = stop_model()
        if result.get("success") and result.get("stopped"):
            print(f"✓ Stopped model: {result.get('name', result.get('stopped'))}")
        print("✓ Server stopped\n")
        server.shutdown()


if __name__ == "__main__":
    main()
