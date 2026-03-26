# D.A.R.A — Dynamic Adaptive Risk Advisor

> *"Demokratisasi manajemen risiko institusional untuk 12,7 juta investor ritel Indonesia."*

**Hackathon Dig Daya × Penguatan Ketahanan dan Inovasi Keuangan 2026**
Kategori: *Rekomendasi Produk Keuangan & Analisis Risiko*

---

## Masalah

Indonesia mencatat pertumbuhan investor ritel **3× lipat** sejak 2019 menjadi
**12,7 juta investor** (OJK, 2024). Namun mayoritas masih rentan terhadap volatilitas:

- IHSG turun **38%** dalam 5 minggu saat COVID-19 (Feb–Mar 2020)
- Investor yang panik jual saat krisis rata-rata merealisasikan kerugian
  **2–3× lebih besar** dari yang mempertahankan posisi terstruktur
- Produk robo-advisor yang ada (Bibit, Bareksa) tidak memberikan
  **analisis risiko real-time yang dipersonalisasi**

### Market Opportunity

| | Angka |
|---|---|
| Total Addressable Market (TAM) | 12,7 juta investor × rata-rata AUM Rp 45 juta = **Rp 571 triliun** |
| Serviceable Addressable Market (SAM) | Segmen digital-savvy 25–45 tahun ≈ 3 juta investor |
| Serviceable Obtainable Market (SOM, Y1) | Target 50.000 pengguna premium |
| Revenue Model | Freemium: Rp 15.000/bulan · Enterprise: 0,1% AUM/tahun |
| Estimasi Revenue Y1 | 50.000 × Rp 15.000 × 12 = **Rp 9 miliar/tahun** |

---

## Solusi: D.A.R.A

Tiga lapisan analisis yang berjalan secara **real-time tanpa refresh browser**:

```
Lapisan 1 — Risiko Kuantitatif
  GARCH(1,1) MLE-fitted → CVaR 95%/99% → volatilitas kondisional

Lapisan 2 — Sentimen Berita AI
  NewsAPI → TF-IDF keyword → FinBERT → Gaussian Mixture Model (Bull/Bear)

Lapisan 3 — Optimasi Portofolio
  Multi-Factor Regime Classifier (4 sinyal) → Hierarchical Risk Parity

  → Rekomendasi alokasi reksa dana dalam Rupiah, push otomatis via SSE
```

---

## Hasil Terukur (Backtesting + Out-of-Sample Validation)

Simulasi 5 tahun dengan **pemisahan train/test** (70% / 30%):

| Metrik | IHSG Buy & Hold | D.A.R.A Adaptive |
|--------|:-:|:-:|
| Volatilitas Tahunan | 16,8% | **5,2%** |
| **Reduksi Volatilitas (Out-of-Sample)** | — | **73%** |
| Return saat Krisis Simulasi | −8,8% | **−2,8%** |
| Proteksi Krisis | — | **+6,0 pp** |

> OOS Period: Oktober 2023 – Maret 2025 · Bukan jaminan hasil masa depan.

**Contoh konkret:**
> Investor dengan modal Rp 100 juta hanya mengalami kerugian Rp 2,8 juta
> saat krisis, dibanding Rp 8,8 juta tanpa D.A.R.A — **selisih Rp 6 juta
> dari satu kejadian krisis saja.**

---

## Keunggulan vs Kompetitor

| Fitur | D.A.R.A | Bibit | Bareksa | Ajaib |
|-------|:-------:|:-----:|:-------:|:-----:|
| GARCH MLE-fitted real-time | ✅ | ❌ | ❌ | ❌ |
| Sentimen FinBERT + GMM | ✅ | ❌ | ❌ | ❌ |
| CVaR dalam nominal Rupiah | ✅ | ❌ | ❌ | ❌ |
| "Cuaca Pasar" visual intuitif | ✅ | ❌ | ❌ | ❌ |
| Real-time tanpa refresh (SSE) | ✅ | ❌ | ❌ | ❌ |
| Out-of-sample backtesting | ✅ | ❌ | ❌ | ❌ |
| Keyword discovery otomatis | ✅ | ❌ | ❌ | ❌ |
| Dual-view (ritel + teknikal) | ✅ | ❌ | ❌ | ❌ |

---

## Arsitektur Teknis

```
keyword_discovery.py
  TF-IDF + korelasi IHSG → query_keywords.json
          ↓
sentiment_analysis.py          demo_sentiment.py
  NewsAPI + FinBERT + GMM       RSS + rule-based (no API key)
          ↓   (priority 1)      ↓   (priority 2 / fallback)
          └──────── SentimentBridge ──────────────────────────┐
                                                              ↓
dara_week1.py                        dara_week3.py
  GARCH(1,1) MLE-fitted   ─────────► RiskRegimeClassifier
  CVaR 95% / 99%                     HRPOptimizer (from scratch)
  5 aset historis                    run_pipeline()
          ↓                                  ↓
                    dara_server.py  (Flask)
                    ┌─────────────────────────────────────┐
                    │ /api/stream    → SSE push 30 detik  │
                    │ /api/analyze   → one-shot JSON       │
                    │ /api/backtest  → OOS simulation      │
                    │ /api/health    → status komponen     │
                    └─────────────────────────────────────┘
                                    ↓
              dara_week4.html + dara_week4.css + dara_week4.js
                    EventSource → auto-update tanpa refresh
```

---

## Instalasi & Menjalankan

### Prasyarat
- Python 3.10+

### Langkah

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. (Opsional) Konfigurasi API key NewsAPI di .env
#    NEWS_API_KEY=your_key_here

# 3. (Opsional) Jalankan keyword discovery
python keyword_discovery.py

# 4. Demo satu klik — lihat output pipeline di terminal
python demo.py

# 5. Jalankan server
python dara_server.py

# 6. Buka browser
#    http://localhost:5000         → UI Ritel
#    http://localhost:5000/classic → UI Teknikal
```

### Mode Operasi

| Kondisi | Perilaku |
|---------|----------|
| Tanpa internet | Simulasi historis 5 tahun (fat-tails, volatility clustering) |
| Dengan yfinance | Data IHSG/SBN live dari Yahoo Finance |
| Dengan NEWS_API_KEY | FinBERT pipeline → sentimen real-time |
| Tanpa API key | RSS rule-based → sentimen dari Google/Reuters RSS |

---

## API Endpoints

| Endpoint | Deskripsi |
|----------|-----------|
| `GET /` | UI Ritel (dara_week4.html) |
| `GET /classic` | UI Teknikal (dara_dashboard.html) |
| `GET /api/stream?scenario=X` | **SSE real-time** — push tiap 30 detik |
| `GET /api/analyze?scenario=X` | One-shot JSON pipeline |
| `GET /api/backtest` | Simulasi historis + OOS validation |
| `GET /api/health` | Status engine, GARCH, sentiment, SSE clients |
| `GET /api/sensitivity` | Semua skenario sekaligus |
| `GET /api/refresh-sentiment` | Paksa refresh FinBERT atau RSS |
| `GET /api/refresh-keywords` | Paksa refresh keyword discovery |
| `GET /api/sentiment-status` | Status cache + keywords aktif |

**Skenario valid:** `crisis` · `bearish` · `moderate` · `bullish` · `euphoria`

---

## Roadmap Komersialisasi

| Fase | Fitur | Target    |
|------|-------|-----------|
| **MVP** | Core engine + SSE real-time + backtesting OOS | ✅ Selesai |
| **v1.1** | Integrasi NAV reksa dana live via OJK API | Q1 2026   |
| **v1.2** | Notifikasi push WhatsApp/Telegram saat regime berubah | Q3 2026   |
| **v2.0** | White-label SDK untuk platform sekuritas (BRI, Mandiri Investasi) | Q1 2027   |
| **v3.0** | Personalisasi profil risiko OJK + machine learning on user behavior | 2027      |

---

## Disclaimer

D.A.R.A adalah *decision-support tool*, bukan layanan investasi berlisensi OJK.
Dirancang sesuai semangat **Sandbox Inovasi OJK** untuk mendorong literasi
keuangan digital. Investasi mengandung risiko kehilangan modal.

---

*Hackathon Dig Daya 2026*
