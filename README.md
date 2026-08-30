# Prompt Lens

A Streamlit dashboard that shows **how** a model decided, not only **what** it said. Each prompt returns four JSON layers — intent, safety, reasoning, and the final answer — rendered as cards. If Groq is missing, rate-limited, or returns bad JSON, a local mock still drives the same UI.

## Run locally (about 3 minutes)

1. Get a free key at [console.groq.com](https://console.groq.com).
2. From this folder:

```bash
pip install -r requirements.txt
```

3. Set the key (PowerShell):

```powershell
$env:GROQ_API_KEY = "your_key_here"
```

Or copy `.env.example` to `.env` and load it yourself. You can also paste the key in the sidebar (session only; it is never written to disk).

4. Start the app:

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). Click a sample prompt, then **Reveal thinking**.

No key? Leave it blank or turn on **Force mock** in the sidebar. The four cards still populate.

## Streamlit Community Cloud secrets

This app is native Python Streamlit — deploy it on [Streamlit Community Cloud](https://share.streamlit.io), not Vercel.

1. Push this repo to GitHub.
2. Create an app pointing at `app.py`.
3. In the Cloud dashboard go to **App settings → Secrets** and add:

```toml
GROQ_API_KEY = "your_key_here"
```

The app reads `st.secrets["GROQ_API_KEY"]` first, then the environment variable, then the sidebar field.

## Demo path

1. Sample prompt → four cards fill (happy path).
2. Toggle **Force mock** → same UI, labeled `MOCK`.
3. Use the privacy/borderline sample → risk and boundary decision change.
4. Optional: clear the key to prove `FALLBACK`.

Default live model: `openai/gpt-oss-120b` (Groq’s replacement after Llama 3.3 retired on free/developer plans, 16 Aug 2026). Use `openai/gpt-oss-20b` if you hit rate limits, or `qwen/qwen3.6-27b` as an extra option.
