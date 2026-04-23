# Hybrid Pipeline Demo
### Narrative News → Procedural Text Conversion

**Tugas Akhir** — Lathifah Sahda · NRP 5025221159  
Teknik Informatika ITS · 2026  
Pembimbing: Shintami Chusnul Hidayati, S.Kom., M.Sc., Ph.D.

---

## 📋 Deskripsi

Demo komparatif tiga model LLM (Groq, GPT, Claude) untuk konversi teks naratif berita menjadi format prosedural menggunakan constraint-based hybrid pipeline.

## 🔧 Struktur Pipeline

```
Fase 2: Rule-Based Pre-Processing
  └── Text cleaning → NER → Action Verbs → Temporal Markers → Constraints

Fase 3: Constrained LLM Conversion
  └── Groq (Llama 3.3 70B) | GPT (gpt-4o-mini) | Claude (Haiku)

Fase 4: Rule-Based Post-Processing
  └── Fix Numbering → Validate Format → Validate Entities → Quality Score

Fase 5: Evaluasi
  └── ROUGE-1/2/L + BERTScore F1 + Entity Preservation Rate
```

## 🚀 Setup & Run

### 1. Clone & Install
```bash
git clone <repo-url>
cd hybrid-pipeline-demo
python -m venv venv
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Konfigurasi API Keys
```bash
cp .env.example .env
# Edit .env dan isi API keys
```

### 3. Jalankan App
```bash
streamlit run app.py
```

Buka browser di: http://localhost:8501

## 🌐 Deploy ke Streamlit Community Cloud (Gratis)

1. Push ke GitHub (pastikan `.env` ada di `.gitignore`)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect repo GitHub kamu
4. Di bagian **Secrets**, tambahkan:
```toml
GROQ_API_KEY = "your_key"
OPENAI_API_KEY = "your_key"
ANTHROPIC_API_KEY = "your_key"
```
5. Deploy!

## 📊 Metrik Evaluasi

| Metrik | Library | Target |
|--------|---------|--------|
| ROUGE-1/2/L | rouge-score | ROUGE-L > 0.4 |
| BERTScore F1 | bert-score | > 0.7 |
| Entity Preservation | Custom | > 85% |
| Quality Score | Custom | Composite |

## 🔑 Cara Mendapatkan API Keys (Gratis)

- **Groq**: [console.groq.com](https://console.groq.com) — gratis, unlimited untuk Llama 3.3 70B
- **OpenAI**: [platform.openai.com](https://platform.openai.com) — perlu credits
- **Anthropic**: [console.anthropic.com](https://console.anthropic.com) — perlu credits
