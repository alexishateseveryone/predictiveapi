# Render Python API (Flask)

This folder contains a minimal Flask API ready to deploy on Render and be consumed by a static website (e.g., hosted on InfinityFree).

## Files
- `app.py` – Flask app with CORS support; reads `ALLOWED_ORIGINS` env var
- `runner_adapter.py` – Calls `bsit_runner`/`bsit_recommendation` functions (`predict`/`run`/`main`)
- `requirements.txt` – Python dependencies
- `runtime.txt` – Python runtime version
- `.gitignore` – Ignores envs, caches, IDE files
 - `models/rf_ict.pkl` – Place your model here (create the `models` folder)

## Local run (Windows PowerShell)
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:ALLOWED_ORIGINS = "http://localhost:5500,https://localhost"
$env:MODEL_PATH = (Resolve-Path .\models\rf_ict.pkl)
python app.py
```
Visit `http://127.0.0.1:5000/health`.

## Deploy to Render
1. Push this folder to a GitHub repo.
2. In Render: New → Web Service → connect the repo.
3. Settings:
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2`
   - Region: closest to you
   - Plan: Free
4. Add Environment Variable:
   - `ALLOWED_ORIGINS` → `https://YOUR_INFINITYFREE_DOMAIN,http://YOUR_INFINITYFREE_DOMAIN`
     - Example: `https://yourname.epizy.com,http://yourname.epizy.com`
5. Deploy and note your public URL, e.g. `https://your-service.onrender.com`.

### Put your files in place
- Copy `bsit_runner.py` and/or `bsit_recommendation.py` into this same folder (next to `app.py`).
- Create a folder `models` and put `rf_ict.pkl` inside it, or set env var `MODEL_PATH` in Render to your model path.

### How the adapter calls your code
- Endpoint `POST /api/recommend` reads JSON and passes it as a single argument to the first available function it finds in your modules, in this order: `predict`, then `run`, then `main`.
- Make sure your function accepts one parameter (a dict) and returns JSON-serializable data.

## Front-end example (InfinityFree)
Use this snippet in your site to call the API:
```html
<script>
async function loadGreeting() {
  const apiBase = "https://your-service.onrender.com"; // replace with your Render URL
  const res = await fetch(`${apiBase}/api/hello`);
  const data = await res.json();
  document.getElementById("result").textContent = data.message;
}

document.addEventListener("DOMContentLoaded", loadGreeting);
</script>
<div id="result"></div>
```

## Notes
- Update `ALLOWED_ORIGINS` with your real InfinityFree domain to avoid CORS errors.
- Both HTTP and HTTPS variants are recommended if your InfinityFree site isn’t fully on HTTPS yet.
- Free plans may cold start; first request after idle might be slow.


