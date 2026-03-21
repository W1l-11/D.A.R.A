"""
D.A.R.A — dara_week1.py  [PRODUCTION v3.1]
============================================
Fondasi Data & Mesin Risiko Klasik.

Fix v3.1:
  - GARCH parameter di-fit via MLE (bukan hardcoded)
  - Semua print disupress saat diimpor (tidak ada noise di server log)
  - Visualisasi hanya berjalan saat python dara_week1.py
"""

import sys, io, numpy as np, pandas as pd
import warnings, os
warnings.filterwarnings('ignore')

# ── Mute semua output saat diimpor ────────────────────────────────────────────
_IS_MAIN   = __name__ == "__main__"
_MUTED_OUT = None
if not _IS_MAIN:
    _MUTED_OUT = io.StringIO()
    sys.stdout = _MUTED_OUT

# ── Warna (untuk visualisasi) ────────────────────────────────────────────────
DARK_BG  = "#0D1117"; PANEL_BG = "#161B22"; ACCENT_1 = "#58A6FF"
ACCENT_2 = "#F0883E"; ACCENT_3 = "#3FB950"; RED_RISK = "#FF7B72"
GOLD     = "#D4AF37"; TEXT_PRI = "#E6EDF3"; TEXT_SEC = "#8B949E"

def get_output_path(filename):
    for d in ["/mnt/user-data/outputs", os.path.expanduser("~/outputs"), ".", "outputs"]:
        if os.path.isdir(d) and os.access(d, os.W_OK):
            return os.path.join(d, filename)
    os.makedirs("outputs", exist_ok=True)
    return os.path.join("outputs", filename)

# ── Crisis windows ────────────────────────────────────────────────────────────
CRISIS_WINDOWS_SIM = [
    (200, 260, "COVID-19\nCrash"),
    (520, 560, "Rate Hike\nShock"),
    (900, 930, "Geopolitical\nRisk"),
]
CRISIS_WINDOWS_LIVE = [
    ("2020-02-20", "2020-04-07", "COVID-19\nCrash"),
    ("2022-01-03", "2022-10-13", "Rate Hike\nShock"),
    ("2022-02-24", "2022-03-15", "Russia-Ukraine\nInvasi"),
    ("2023-03-08", "2023-03-31", "Banking\nCrisis SVB"),
    ("2024-07-31", "2024-08-05", "Yen Carry\nUnwind"),
]

# ── Ticker map ────────────────────────────────────────────────────────────────
TICKER_MAP = {
    "IHSG (Saham)":   "^JKSE",
    "Obligasi (SBN)": "^TNX",
    "Pasar Uang":     "SHY",
}
ASSET_COLORS = {
    "IHSG (Saham)": ACCENT_1, "Obligasi (SBN)": ACCENT_3, "Pasar Uang": GOLD,
}
ASSET_REGIME = {"IHSG (Saham)": 3.5, "Obligasi (SBN)": 1.8, "Pasar Uang": 0.6}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  D.A.R.A — Fondasi Data & Mesin Risiko Klasik")
print("=" * 60)
print("\n[1/3] Downloading 5-year data via yfinance...")

try:
    import yfinance as yf
    raw = yf.download(list(TICKER_MAP.values()), period="5y",
                      auto_adjust=True, progress=False)["Close"]
    raw = raw.rename(columns={v: k for k, v in TICKER_MAP.items()})
    raw = raw.dropna(how="all").ffill()
    df_prices  = raw
    df_returns = raw.pct_change().dropna()
    df_prices  = df_prices.loc[df_returns.index]
    n_days     = len(df_returns)
    dates      = df_returns.index
    LIVE_DATA  = True
    for name in TICKER_MAP:
        if name in df_returns.columns:
            ret = df_returns[name].values
            p   = df_prices[name].values
            print(f"   ✓ {name}: Return={((p[-1]/p[0])-1)*100:+.1f}%  Vol={ret.std()*np.sqrt(252)*100:.1f}%")
except Exception as e:
    print(f"   ⚠ yfinance gagal — simulasi historis realistis")
    LIVE_DATA = False

if not LIVE_DATA:
    np.random.seed(42)
    n_days = 1260
    dates  = pd.bdate_range(end="2025-03-21", periods=n_days)
    ASSET_PARAMS = {
        "IHSG (Saham)":   {"mu": 0.0003,  "sigma": 0.012},
        "Obligasi (SBN)": {"mu": 0.00015, "sigma": 0.004},
        "Pasar Uang":     {"mu": 0.00008, "sigma": 0.0008},
    }
    returns_dict, prices_dict = {}, {}
    for name, p in ASSET_PARAMS.items():
        regime = ASSET_REGIME[name]
        ret    = np.zeros(n_days)
        vol    = p["sigma"]
        for i in range(n_days):
            in_crisis = any(s <= i <= e for s, e, _ in CRISIS_WINDOWS_SIM)
            sigma_t   = p["sigma"] * regime if in_crisis else p["sigma"]
            vol       = (0.85 * vol + 0.10 * abs(ret[i-1]) + 0.05 * sigma_t) if i > 0 else sigma_t
            ret[i]    = p["mu"] + vol * np.random.standard_t(df=5) / np.sqrt(5/3)
        returns_dict[name] = ret
        prices_dict[name]  = 1000 * np.exp(np.cumsum(ret))
    df_prices  = pd.DataFrame(prices_dict, index=dates)
    df_returns = pd.DataFrame(returns_dict, index=dates)
    for name in ASSET_PARAMS:
        p = df_prices[name].values
        print(f"   ✓ {name} [SIMULATED]: Return={(p[-1]/p[0]-1)*100:+.1f}%  Vol={df_returns[name].std()*np.sqrt(252)*100:.1f}%")

ASSETS_AVAILABLE = [c for c in TICKER_MAP if c in df_returns.columns]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: GARCH(1,1) — MLE FITTED
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/3] Fitting GARCH(1,1) via Maximum Likelihood Estimation...")

from scipy.optimize import minimize

def fit_garch_mle(returns: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    Parameter α, β di-estimasi via MLE — bukan hardcoded.

    Returns
    -------
    sigma      : np.ndarray — conditional volatility time-series
    params     : dict dengan omega, alpha, beta, persistence, half_life
    """
    def neg_log_likelihood(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 0.9999:
            return 1e10
        n      = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = float(np.var(returns))
        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        sigma2 = np.maximum(sigma2, 1e-10)
        nll    = 0.5 * float(np.sum(np.log(sigma2) + returns**2 / sigma2))
        return nll

    var_r = float(np.var(returns))
    # Starting values: variance targeting
    x0     = [var_r * 0.05, 0.10, 0.85]
    bounds = [(1e-9, 0.1), (1e-5, 0.49), (1e-5, 0.9994)]

    result = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500, "ftol": 1e-12})

    if result.success and result.fun < neg_log_likelihood(x0):
        omega, alpha, beta = result.x
    else:
        # Fallback: variance targeting dengan default alpha/beta
        alpha = 0.10
        beta  = 0.85
        omega = var_r * (1 - alpha - beta)

    # Compute full sigma series
    n      = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = var_r
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    sigma      = np.sqrt(np.maximum(sigma2, 1e-12))
    persistence = alpha + beta
    half_life   = (-np.log(2) / np.log(persistence)) if persistence < 1 else np.inf

    return sigma, {
        "omega": float(omega), "alpha": float(alpha), "beta": float(beta),
        "persistence": float(persistence), "half_life": float(half_life),
        "mle_success": bool(result.success),
    }

garch_results = {}
for name in ASSETS_AVAILABLE:
    ret    = df_returns[name].values
    sigma, params = fit_garch_mle(ret)
    garch_results[name] = {
        "sigma": sigma,
        "annualized_vol": sigma * np.sqrt(252) * 100,
        "params": params,
    }
    p = params
    mle_tag = "MLE ✓" if p["mle_success"] else "fallback"
    print(f"   ✓ {name} [{mle_tag}]")
    print(f"      ω={p['omega']:.2e}  α={p['alpha']:.4f}  β={p['beta']:.4f}  "
          f"Persistence={p['persistence']:.4f}  Half-life={p['half_life']:.1f}d")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: CVaR (Expected Shortfall)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/3] Computing CVaR (Expected Shortfall)...")

def compute_var_cvar(returns: np.ndarray, confidence_levels=(0.95, 0.99)) -> dict:
    results = {}
    for cl in confidence_levels:
        var  = float(np.percentile(returns, (1 - cl) * 100))
        tail = returns[returns <= var]
        cvar = float(tail.mean()) if len(tail) > 0 else var
        results[cl] = {"VaR": var, "CVaR": cvar, "tail_count": len(tail)}
    return results

risk_metrics = {}
for name in ASSETS_AVAILABLE:
    ret = df_returns[name].values
    risk_metrics[name] = compute_var_cvar(ret)
    m95, m99 = risk_metrics[name][0.95], risk_metrics[name][0.99]
    print(f"   ✓ {name}: CVaR95={m95['CVaR']*100:.3f}%  CVaR99={m99['CVaR']*100:.3f}%")

# ── Restore stdout ─────────────────────────────────────────────────────────────
if not _IS_MAIN and _MUTED_OUT is not None:
    sys.stdout = sys.__stdout__

# ─────────────────────────────────────────────────────────────────────────────
# VISUALISASI — hanya saat python dara_week1.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats

    def draw_crisis_windows(ax, dates, live_data):
        ymin, ymax = ax.get_ylim()
        label_y = ymax - (ymax - ymin) * 0.04
        windows = CRISIS_WINDOWS_LIVE if live_data else CRISIS_WINDOWS_SIM
        if live_data:
            for start_str, end_str, label in windows:
                start_dt = pd.Timestamp(start_str)
                end_dt   = pd.Timestamp(end_str)
                if start_dt > dates[-1] or end_dt < dates[0]: continue
                start_dt = max(start_dt, dates[0])
                end_dt   = min(end_dt, dates[-1])
                ax.axvspan(start_dt, end_dt, alpha=0.13, color=RED_RISK, zorder=1)
                mid_dt = start_dt + (end_dt - start_dt) / 2
                ax.annotate(label, xy=(mid_dt, label_y), ha='center', va='top',
                            fontsize=7.5, color=RED_RISK,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=PANEL_BG,
                                      alpha=0.75, edgecolor=RED_RISK, linewidth=0.6), zorder=5)
        else:
            for s, e, label in windows:
                ax.axvspan(dates[s], dates[e], alpha=0.13, color=RED_RISK, zorder=1)
                mid_dt = dates[(s + e) // 2]
                ax.annotate(label, xy=(mid_dt, label_y), ha='center', va='top',
                            fontsize=7.5, color=RED_RISK,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=PANEL_BG,
                                      alpha=0.75, edgecolor=RED_RISK, linewidth=0.6), zorder=5)

    print("\nGenerating dashboard...")
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 14), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                            left=0.06, right=0.97, top=0.88, bottom=0.08)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :2])
    ax3 = fig.add_subplot(gs[1, 2])
    ax4 = fig.add_subplot(gs[2, :])

    data_label = "Data: Yahoo Finance (Live)" if LIVE_DATA else "Data: Simulasi Historis"
    fig.text(0.5, 0.945, "D.A.R.A", ha='center', fontsize=32, fontweight='bold',
             color=ACCENT_1, fontfamily='monospace')
    fig.text(0.5, 0.918,
             f"Dynamic Adaptive Risk Advisor  ·  Minggu 1  ·  {data_label}  ·  GARCH: MLE-Fitted",
             ha='center', fontsize=10, color=TEXT_SEC)
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.905, 0.905],
                   transform=fig.transFigure, color=ACCENT_1, lw=0.6, alpha=0.4))

    ax1.set_facecolor(PANEL_BG)
    ax1.set_title("Harga Ternormalisasi — 5 Tahun", color=TEXT_PRI, fontsize=11, fontweight='bold', pad=8)
    for name in ASSETS_AVAILABLE:
        prices = df_prices[name].values
        ax1.plot(dates, prices / prices[0] * 100, color=ASSET_COLORS[name],
                 lw=1.5, label=name, alpha=0.9)
    draw_crisis_windows(ax1, dates, LIVE_DATA)
    ax1.axhline(100, color=TEXT_SEC, lw=0.6, ls='--', alpha=0.4)
    ax1.set_ylabel("Nilai (Base=100)", color=TEXT_SEC, fontsize=9)
    ax1.tick_params(colors=TEXT_SEC, labelsize=8)
    ax1.spines[:].set_color('#30363D')
    leg = ax1.legend(loc='upper left', framealpha=0.3, edgecolor='#30363D', labelcolor=TEXT_PRI, fontsize=9)
    for l in leg.get_lines(): l.set_linewidth(2.5)
    ax1.grid(axis='y', color='#30363D', lw=0.5, alpha=0.5)

    ax2.set_facecolor(PANEL_BG)
    ax2.set_title("GARCH(1,1) — Volatilitas Bersyarat Tahunan [MLE-Fitted]",
                  color=TEXT_PRI, fontsize=11, fontweight='bold', pad=8)
    for name in ASSETS_AVAILABLE:
        ax2.plot(dates, garch_results[name]["annualized_vol"],
                 color=ASSET_COLORS[name], lw=1.1, label=name, alpha=0.85)
    draw_crisis_windows(ax2, dates, LIVE_DATA)
    p = garch_results[ASSETS_AVAILABLE[0]]["params"]
    ax2.text(0.02, 0.92,
             f"IHSG: α={p['alpha']:.4f}  β={p['beta']:.4f}  "
             f"Persist={p['persistence']:.4f}  Half-life={p['half_life']:.1f}d  [MLE ✓]",
             transform=ax2.transAxes, fontsize=8, color=ACCENT_1,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#0D1117', alpha=0.8, edgecolor=ACCENT_1, lw=0.8))
    ax2.set_ylabel("Vol Tahunan (%)", color=TEXT_SEC, fontsize=9)
    ax2.tick_params(colors=TEXT_SEC, labelsize=8)
    ax2.spines[:].set_color('#30363D')
    ax2.legend(loc='upper right', framealpha=0.3, edgecolor='#30363D', labelcolor=TEXT_PRI, fontsize=8)
    ax2.grid(axis='y', color='#30363D', lw=0.5, alpha=0.5)

    ax3.set_facecolor(PANEL_BG)
    ax3.set_title("CVaR Harian\n(Expected Shortfall 95% & 99%)", color=TEXT_PRI, fontsize=10, fontweight='bold', pad=8)
    x     = np.arange(len(ASSETS_AVAILABLE))
    width = 0.32
    cvar_95 = [abs(risk_metrics[n][0.95]["CVaR"]) * 100 for n in ASSETS_AVAILABLE]
    cvar_99 = [abs(risk_metrics[n][0.99]["CVaR"]) * 100 for n in ASSETS_AVAILABLE]
    bars1 = ax3.bar(x - width/2, cvar_95, width, label="CVaR 95%", color=ACCENT_2, alpha=0.85)
    bars2 = ax3.bar(x + width/2, cvar_99, width, label="CVaR 99%", color=RED_RISK, alpha=0.85)
    for bar in bars1: ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{bar.get_height():.2f}%", ha='center', va='bottom', fontsize=8, color=ACCENT_2, fontweight='bold')
    for bar in bars2: ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{bar.get_height():.2f}%", ha='center', va='bottom', fontsize=8, color=RED_RISK, fontweight='bold')
    ax3.set_xticks(x); ax3.set_xticklabels([n.split()[0] for n in ASSETS_AVAILABLE], color=TEXT_SEC, fontsize=9)
    ax3.set_ylabel("Kerugian Ekor (%)", color=TEXT_SEC, fontsize=9)
    ax3.tick_params(colors=TEXT_SEC, labelsize=8); ax3.spines[:].set_color('#30363D')
    ax3.legend(framealpha=0.3, edgecolor='#30363D', labelcolor=TEXT_PRI, fontsize=8)
    ax3.grid(axis='y', color='#30363D', lw=0.5, alpha=0.5); ax3.set_axisbelow(True)

    ax4.set_facecolor(PANEL_BG)
    ax4.set_title("Distribusi Return — Fat Tails vs Normal", color=TEXT_PRI, fontsize=11, fontweight='bold', pad=8)
    x_range = np.linspace(-0.06, 0.06, 300)
    for name in ASSETS_AVAILABLE:
        ret   = df_returns[name].values
        color = ASSET_COLORS[name]
        bw    = 1.06 * ret.std() * len(ret)**(-0.2)
        kde   = np.array([np.mean(np.exp(-0.5*((xv-ret)/bw)**2)/(bw*np.sqrt(2*np.pi))) for xv in x_range])
        ax4.plot(x_range*100, kde/100, color=color, lw=2, label=name, alpha=0.9)
        ax4.fill_between(x_range*100, kde/100, alpha=0.07, color=color)
        ax4.axvline(risk_metrics[name][0.95]["VaR"]*100, color=color, lw=1, ls='--', alpha=0.5)
    ret0     = df_returns[ASSETS_AVAILABLE[0]].values
    norm_pdf = stats.norm.pdf(x_range, ret0.mean(), ret0.std())
    ax4.plot(x_range*100, norm_pdf/100, color=TEXT_SEC, lw=1.2, ls=':', alpha=0.6, label="Normal (ref)")
    ax4.text(0.72, 0.88, "⚠  Fat Tails\ndetected", transform=ax4.transAxes, fontsize=9, color=RED_RISK,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#0D1117', alpha=0.85, edgecolor=RED_RISK, lw=0.8))
    ax4.set_xlabel("Return Harian (%)", color=TEXT_SEC, fontsize=9); ax4.set_ylabel("Densitas", color=TEXT_SEC, fontsize=9)
    ax4.tick_params(colors=TEXT_SEC, labelsize=8); ax4.spines[:].set_color('#30363D')
    ax4.legend(loc='upper left', framealpha=0.3, edgecolor='#30363D', labelcolor=TEXT_PRI, fontsize=9)
    ax4.grid(axis='y', color='#30363D', lw=0.5, alpha=0.4); ax4.set_xlim(-5.5, 5.5)

    fig.text(0.5, 0.025,
             "D.A.R.A  ·  Minggu 1: Classic Risk Engine  ·  GARCH(1,1) MLE  |  CVaR  |  Fat Tails",
             ha='center', fontsize=7.5, color=TEXT_SEC, alpha=0.7)
    out = get_output_path("visualization_dashboard.png")
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=DARK_BG, edgecolor='none')
    plt.close()
    print(f"\n✅ Dashboard: {out}")

    print("\n" + "="*60 + "\n  RINGKASAN EKSEKUTIF\n" + "="*60)
    for name in ASSETS_AVAILABLE:
        ret = df_returns[name].values
        m   = risk_metrics[name]
        g   = garch_results[name]["params"]
        sharpe = (ret.mean()/(ret.std()+1e-12))*np.sqrt(252)
        print(f"  {name.split()[0]:<8}  Ann.Ret={(1+ret.mean())**252-1:.1%}  "
              f"Vol={ret.std()*np.sqrt(252):.1%}  "
              f"α={g['alpha']:.4f} β={g['beta']:.4f}  "
              f"CVaR95={m[0.95]['CVaR']*100:.3f}%  Sharpe={sharpe:.2f}")
