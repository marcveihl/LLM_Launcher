# LLM Launcher v2

<img width="896" height="630" alt="image" src="https://github.com/user-attachments/assets/3af87c8b-a016-447e-a3b5-c2d466eb3fec" />

Remote control for local LLM models with authentication, monitoring, and Tailscale support.

## What's New in v2

- **API Key Authentication** - Secure access from anywhere
- **GPU/System Monitoring** - Real-time VRAM, GPU usage, and temp
- **Health Checks** - Shows when model is actually ready for inference
- **Live Log Tailing** - Watch llama-server output in the UI
- **External Config** - Easy model management via JSON
- **Tailscale Ready** - Auto-detects Tailscale IP

---

## Quick Setup Guide

### Prerequisites

- **Python 3.8+** - [Download from python.org](https://www.python.org/downloads/)
- **NVIDIA GPU** (recommended) - For GPU acceleration
- **CMake** and **Visual Studio Build Tools** (Windows) - For building llama.cpp

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Build llama.cpp with llama-server

**Windows:**
```powershell
# Clone llama.cpp
cd C:\LLM_Tools
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with CUDA support (NVIDIA GPU)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# llama-server.exe will be at:
# C:\LLM_Tools\llama.cpp\build\bin\Release\llama-server.exe
```

**Linux/Mac:**
```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with CUDA (Linux) or Metal (Mac)
make GGML_CUDA=1  # Linux with NVIDIA GPU
make              # Mac with Metal

# llama-server will be at: ./llama-server
```

### Step 3: Download GGUF Models

Download quantized GGUF models from [Hugging Face](https://huggingface.co/models?library=gguf).

**Recommended sources:**
- **LM Studio Models Directory**: `~/.lmstudio/models` or `%USERPROFILE%\.lmstudio\models`
- **Manual Download**: Create a models folder (e.g., `C:\LLM_Tools\models`)

**Example models:**
```bash
# Using huggingface-cli
pip install huggingface-hub
huggingface-cli download unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF \
  Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
  --local-dir C:\LLM_Tools\models\qwen3
```

Or download directly from browser:
- [Qwen Models](https://huggingface.co/models?search=qwen%20gguf)
- [Llama Models](https://huggingface.co/models?search=llama%20gguf)

### Step 4: Configure LLM Launcher

1. **Copy the example config:**
   ```bash
   cp config.json.example config.json
   ```

2. **Edit `config.json`** with your paths:

   ```json
   {
     "server": {
       "host": "0.0.0.0",
       "port": 8081,
       "llama_host": "0.0.0.0",
       "llama_port": 8080
     },
     "security": {
       "api_key": "YOUR_SECURE_RANDOM_KEY",
       "allowed_origins": ["*"]
     },
     "paths": {
       "llama_server": "C:\\LLM_Tools\\llama.cpp\\build\\bin\\Release\\llama-server.exe",
       "models_base": "C:\\LLM_Tools\\models"
     },
     "models": {
       "qwen3-30b": {
         "name": "Qwen3 30B Instruct",
         "file": "qwen3/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
         "context": 8192,
         "gpu_layers": 48,
         "temp": 0.7,
         "top_k": 40,
         "top_p": 0.95
       }
     }
   }
   ```

3. **Generate a secure API key:**
   ```powershell
   # PowerShell
   -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 32 | % {[char]$_})
   ```

   ```bash
   # Linux/Mac
   openssl rand -base64 32
   ```

### Step 5: Run the Server

**Windows:**
```powershell
# Double-click start_server.bat, or:
python llm_control_server.py
```

**Linux/Mac:**
```bash
python llm_control_server.py
```

### Step 6: Access the UI

Open your browser to:
- **Local**: `http://localhost:8081`
- **Network**: `http://YOUR_PC_IP:8081`

Enter your API key when prompted, then select a model to start!

---

## Tailscale Setup

Tailscale creates a secure mesh VPN so you can access your PC from anywhere.

### Install Tailscale

**Windows (Gaming PC):**
1. Download from https://tailscale.com/download
2. Install and sign in
3. Note your Tailscale IP (starts with `100.x.x.x`)

**macOS:**
```bash
brew install tailscale
# Or download from App Store / website
```

### Configure for Tailscale

The server auto-detects your Tailscale IP. Just make sure:

1. **llama-server binds to all interfaces** - Already configured in v2:
   ```json
   "llama_host": "0.0.0.0"
   ```

2. **Firewall allows Tailscale** - Usually automatic, but verify:
   - Windows Firewall should show Tailscale as allowed
   - Port 8080 (llama) and 8081 (control) should be accessible

### Access from Anywhere

From your Mac (on Tailscale):
```
http://100.x.x.x:8081      # Control UI
http://100.x.x.x:8080/v1   # OpenAI-compatible API
```

You can also use your Tailscale hostname:
```
http://your-pc-name:8081
```

### Using with External Tools

Once running, point any OpenAI-compatible client to your Tailscale IP:

```python
# Python example
from openai import OpenAI

client = OpenAI(
    base_url="http://100.x.x.x:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="qwen3-coder-30b",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## Configuration Reference

### config.json Structure

```json
{
  "server": {
    "host": "0.0.0.0",        // Control server bind address
    "port": 8081,              // Control server port
    "llama_host": "0.0.0.0",   // llama-server bind address
    "llama_port": 8080         // llama-server port
  },
  "security": {
    "api_key": "...",          // Required for all API calls
    "allowed_origins": ["*"]   // CORS origins (for web UI)
  },
  "paths": {
    "llama_server": "C:\\...", // Path to llama-server.exe
    "models_base": "C:\\..."   // Base path for model files
  },
  "models": {
    "model-id": {
      "name": "Display Name",
      "file": "relative/path/to/model.gguf",
      "context": 8192,
      "gpu_layers": 48,
      "cpu_moe": 24,
      "temp": 0.6,
      "top_k": 20,
      "top_p": 0.95,
      "min_p": 0.0,
      "extra_args": ["--jinja"]  // Additional llama-server args
    }
  }
}
```

### Model Configuration Parameters

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `name` | Display name in UI | Any descriptive name |
| `file` | Path to .gguf file relative to `models_base` | `vendor/model-name.gguf` |
| `context` | Context window size | `8192`, `32768`, `131072` |
| `gpu_layers` | Number of layers to offload to GPU | `0` (CPU only) to `99` (all layers) |
| `cpu_moe` | CPU threads for MoE models | `4` to `24` (based on CPU) |
| `temp` | Temperature (creativity) | `0.1` (focused) to `1.0` (creative) |
| `top_k` | Top-K sampling | `20` to `50` |
| `top_p` | Top-P (nucleus) sampling | `0.9` to `0.95` |
| `min_p` | Minimum probability threshold | `0.0` to `0.1` |
| `extra_args` | Additional llama-server flags | `["--jinja"]`, `["--flash-attn"]` |

**Tips:**
- **VRAM Management**: Reduce `gpu_layers` if you run out of VRAM
- **Context Size**: Larger contexts use more VRAM/RAM
- **Quantization**: Q4_K_M is a good balance of quality and speed
- **MoE Models**: Set `cpu_moe` for Mixtral/Qwen-MoE models

### Adding New Models

1. Download a GGUF model file
2. Add entry to `models` section in `config.json`
3. Restart the control server
4. New model appears in UI

**Example:**
```json
"my-model-id": {
  "name": "My Custom Model",
  "file": "path/to/model.gguf",
  "context": 8192,
  "gpu_layers": 48
}
```

---

## API Reference

All endpoints except `/` require `X-API-Key` header.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| GET | `/api/status` | Current model status |
| GET | `/api/models` | List available models |
| GET | `/api/stats` | GPU/system stats |
| GET | `/api/logs?lines=N` | Recent log lines |
| GET | `/api/network` | Network addresses |
| POST | `/api/start/{model-id}` | Start a model |
| POST | `/api/stop` | Stop current model |

### curl Examples

```bash
# Set your key
KEY="your-api-key"
HOST="100.x.x.x:8081"

# Check status
curl -H "X-API-Key: $KEY" http://$HOST/api/status

# Start a model
curl -X POST -H "X-API-Key: $KEY" http://$HOST/api/start/qwen3-coder-30b

# Stop model
curl -X POST -H "X-API-Key: $KEY" http://$HOST/api/stop

# Get GPU stats
curl -H "X-API-Key: $KEY" http://$HOST/api/stats
```

---

## Security Notes

1. **Change the default API key** - The default key is intentionally obvious
2. **Tailscale provides encryption** - Traffic between devices is encrypted
3. **Don't expose to public internet** - Use Tailscale, not port forwarding
4. **API key stored in browser** - Uses localStorage, clear if on shared device

---

## Troubleshooting

### "llama-server not found"
- Check `paths.llama_server` in config.json
- Ensure llama.cpp is built

### Model won't start
- Check the Logs panel in UI
- Verify model file path exists
- Check GPU has enough VRAM

### Can't connect from Mac
- Verify Tailscale is running on both devices
- Check Windows Firewall allows Python/ports 8080-8081
- Try `ping 100.x.x.x` to verify connectivity

### Health shows "Loading" for too long
- Large models take 30-60s to load
- Check logs for errors
- Reduce `gpu_layers` if running out of VRAM
