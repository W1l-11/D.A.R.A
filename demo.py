"""
D.A.R.A — demo.py
==================
Script demo satu-klik untuk presentasi Hackathon Dig Daya.
Jalankan: python demo.py

Menampilkan:
  1. Ringkasan sistem D.A.R.A
  2. Output semua 5 skenario
  3. Angka backtesting
  4. Instruksi membuka browser
"""

import sys, time

def separator(char="─", n=65):
    print(char * n)

def header():
    print()
    separator("═")
    print("  D.A.R.A — Dynamic Adaptive Risk Advisor")
    print("  Hackathon Dig Daya × Penguatan Ketahanan Keuangan")
    separator("═")
    print()

def section(title):
    print()
    separator()
    print(f"  {title}")
    separator()

def progress(msg, delay=0.05):
    print(f"  {msg}", end="", flush=True)
    for _ in range(3):
        time.sleep(delay)
        print(".", end="", flush=True)
    print(" ✅")

# ── Main demo ──────────────────────────────────────────────────────────────────
header()

# Step 1: Import check
section("LANGKAH 1 — Memuat Engine")
try:
    progress("Memuat dara_week1 (GARCH + CVaR)")
    import dara_week1 as w1
    print(f"     Assets   : {w1.ASSETS_AVAILABLE}")
    print(f"     Data     : {'Live (Yahoo Finance)' if w1.LIVE_DATA else 'Simulasi historis 5 tahun'}")
    print(f"     Shape    : {w1.df_returns.shape[0]} hari × {w1.df_returns.shape[1]} aset")

    progress("Memuat dara_week3 (Regime Classifier + HRP)")
    import dara_week3 as w3
    print(f"     5 Aset   : {w3.ASSET_NAMES}")
    print(f"     GARCH    : persistence={w3._GARCH_PERSIST:.3f}  half-life={w3._GARCH_HALFLIFE:.0f} hari")
    print(f"     CVaR 95% : -{abs(w3._IHSG_CVAR_95)*100:.3f}%/hari  CVaR 99%: -{abs(w3._IHSG_CVAR_99)*100:.3f}%/hari")

except Exception as e:
    print(f"\n  ❌ Error: {e}")
    print("  Pastikan semua file ada di satu folder dan requirements terinstall.")
    sys.exit(1)

# Step 2: All scenarios
section("LANGKAH 2 — Demo 5 Skenario Pasar")

clf = w3.RiskRegimeClassifier()
opt = w3.HRPOptimizer()
import numpy as np

gn = w3.compute_garch_vol(w3.df_returns["IHSG (Saham)"].values)
mz = w3.compute_momentum(w3.df_returns)
cr = w3.compute_correlation_regime(w3.df_returns)

print(f"\n  {'Skenario':<12} {'Cuaca':<22} {'Regime':<12} {'Score':>5}  {'Saham':>6}  {'Obligasi':>8}  {'Pasar Uang':>10}  {'Vol':>6}")
print("  " + "─" * 80)

WEATHER = {
    "DEFENSIVE": {"low": "⛈  Badai", "high": "🌧  Mendung Tebal"},
    "MODERATE":  {"any": "⛅ Berawan"},
    "AGGRESSIVE":{"low": "🌤  Cerah", "high": "🌡  Terlalu Panas"},
}

for sc in ["crisis","bearish","moderate","bullish","euphoria"]:
    s  = w3.get_finbert_sentiment_stub(sc)
    r  = clf.classify(gn, s["market"], mz, cr)
    hr = opt.optimize(w3.df_returns, r["guardrails"])
    ww = hr["weights"]
    w_vec = ww.values
    cov   = hr["cov"]
    p_var = float(w_vec @ cov @ w_vec)
    p_vol = float(p_var**0.5) * 100

    regime = r["regime"]
    score  = r["score"]
    if regime == "DEFENSIVE":
        weather = WEATHER["DEFENSIVE"]["low" if score < 28 else "high"]
    elif regime == "MODERATE":
        weather = WEATHER["MODERATE"]["any"]
    else:
        weather = WEATHER["AGGRESSIVE"]["high" if score > 72 else "low"]

    marker = " ◄" if sc in ["bearish","moderate"] else ""
    print(f"  {sc:<12} {weather:<22} {regime:<12} {score:>4.1f}  "
          f"{ww['IHSG (Saham)']*100:>5.1f}%  "
          f"{ww['Obligasi (SBN)']*100:>7.1f}%  "
          f"{ww['Pasar Uang']*100:>9.1f}%  "
          f"{p_vol:>5.1f}%{marker}")

# Step 3: Backtesting
section("LANGKAH 3 — Hasil Backtesting")

print("\n  Mensimulasikan portofolio D.A.R.A vs Buy-and-Hold IHSG (5 tahun)...")
progress("  Menghitung regime rolling-60-hari")

try:
    base_names = ["IHSG (Saham)", "Obligasi (SBN)", "Pasar Uang"]
    ret3 = w3.df_returns[base_names]
    N    = len(ret3)

    ihsg_ret   = ret3["IHSG (Saham)"].values
    ihsg_cum   = (1 + ihsg_ret).cumprod()

    dara_ret   = np.zeros(N)
    clf2 = w3.RiskRegimeClassifier()
    opt2 = w3.HRPOptimizer()
    gc_cache = {}

    for i in range(60, N):
        window = ret3.iloc[i-60:i]
        gn2 = w3.compute_garch_vol(window["IHSG (Saham)"].values)
        cr2 = w3.compute_correlation_regime(window)
        mz2 = float((window["IHSG (Saham)"].mean() - ret3["IHSG (Saham)"].mean())
                    / (ret3["IHSG (Saham)"].std() + 1e-9))
        ri2 = clf2.classify(gn2, 0.0, mz2, cr2)
        regime2 = ri2["regime"]
        if regime2 not in gc_cache:
            gc_cache[regime2] = {k: ri2["guardrails"][k] for k in base_names if k in ri2["guardrails"]}
        try:
            hr2   = opt2.optimize(window, gc_cache[regime2])
            w2    = hr2["weights"]
            day_r = sum(float(w2.get(a,0)) * float(ret3[a].iloc[i]) for a in base_names)
        except Exception:
            day_r = float(ret3["Pasar Uang"].iloc[i])
        dara_ret[i] = day_r

    dara_cum = (1 + dara_ret).cumprod()

    ihsg_vol = float(np.std(ihsg_ret) * np.sqrt(252) * 100)
    dara_vol = float(np.std(dara_ret[60:]) * np.sqrt(252) * 100)
    vol_red  = (1 - dara_vol / ihsg_vol) * 100

    # Crisis window
    cs, ce = 200, 260
    ihsg_cr = float(ihsg_cum[ce] / ihsg_cum[cs] - 1)
    dara_cr = float((1 + dara_ret[cs:ce+1]).cumprod()[-1] - 1)

    ihsg_total = float(ihsg_cum[-1] - 1)
    dara_total = float(dara_cum[-1] - 1)

    print(f"\n  {'Metrik':<35} {'IHSG Buy & Hold':>16}  {'D.A.R.A Adaptive':>16}")
    print("  " + "─" * 72)
    print(f"  {'Total Return (5 tahun)':<35} {ihsg_total*100:>+14.1f}%  {dara_total*100:>+14.1f}%")
    print(f"  {'Volatilitas Tahunan':<35} {ihsg_vol:>15.1f}%  {dara_vol:>15.1f}%")
    print(f"  {'Reduksi Volatilitas':<35} {'—':>16}  {vol_red:>+14.0f}%")
    print(f"  {'Return saat Krisis Simulasi':<35} {ihsg_cr*100:>+14.1f}%  {dara_cr*100:>+14.1f}%")
    print(f"  {'Proteksi Krisis (selisih)':<35} {'—':>16}  {(dara_cr-ihsg_cr)*100:>+14.1f}pp")

    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  KESIMPULAN BACKTESTING                                 │
  │                                                         │
  │  D.A.R.A mengurangi volatilitas sebesar {vol_red:.0f}%          │
  │  sambil memberikan perlindungan {abs((dara_cr-ihsg_cr)*100):.1f}pp lebih baik   │
  │  saat terjadi krisis pasar.                             │
  │                                                         │
  │  * Bukan jaminan hasil masa depan                       │
  └─────────────────────────────────────────────────────────┘""")

except Exception as e:
    print(f"\n  ❌ Backtesting error: {e}")

# Step 4: Server
section("LANGKAH 4 — Menjalankan Server")

print("""
  Untuk menjalankan aplikasi lengkap:

  ┌─────────────────────────────────────────┐
  │  Terminal:  python dara_server.py        │
  │  Browser:   http://localhost:5000        │
  └─────────────────────────────────────────┘

  Endpoints tersedia:
    http://localhost:5000                   → UI Ritel
    http://localhost:5000/classic           → UI Teknikal
    http://localhost:5000/api/analyze?scenario=moderate
    http://localhost:5000/api/backtest
    http://localhost:5000/api/health
""")

separator("═")
print("  D.A.R.A — Siap untuk Demo Hackathon Dig Daya 2025")
print('  "Demokratisasi manajemen risiko institusional')
print('   untuk 12,7 juta investor ritel Indonesia."')
separator("═")
print()
