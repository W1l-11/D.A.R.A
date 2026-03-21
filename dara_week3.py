"""
D.A.R.A - Dynamic Adaptive Risk Advisor
dara_week3.py — Mesin Rekomendasi [PRODUCTION]
"""
import numpy as np
import pandas as pd
import warnings, os, json, logging, time
warnings.filterwarnings('ignore')

log = logging.getLogger(__name__)
np.random.seed(42)

DARK_BG  = "#0D1117"; PANEL_BG = "#161B22"; ACCENT_1 = "#58A6FF"
ACCENT_2 = "#F0883E"; ACCENT_3 = "#3FB950"; RED_RISK = "#FF7B72"
GOLD     = "#D4AF37"; PURPLE   = "#BC8CFF"; TEXT_PRI = "#E6EDF3"
TEXT_SEC = "#8B949E"; GRID_COL = "#30363D"

ASSET_NAMES  = ["IHSG (Saham)", "Obligasi (SBN)", "Pasar Uang", "REIT", "Komoditas"]
ASSET_COLORS = {
    "IHSG (Saham)": ACCENT_1, "Obligasi (SBN)": ACCENT_3,
    "Pasar Uang": GOLD, "REIT": PURPLE, "Komoditas": ACCENT_2,
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA  — dara_week1 (live/sim 3 aset) + 2 aset simulasi berkorelasi
# ─────────────────────────────────────────────────────────────────────────────
LIVE_DATA       = False
_IHSG_CVAR_95   = -0.0245
_IHSG_CVAR_99   = -0.0395
_GARCH_PERSIST  = 0.950
_GARCH_HALFLIFE = 14.0
_GARCH_SERIES   = None   # time-series GARCH annualized vol untuk chart

try:
    import dara_week1 as _w1
    _base = _w1.df_returns.copy()
    _N    = len(_base)
    _rng  = np.random.default_rng(99)
    _ihsg = _base["IHSG (Saham)"].values
    _base["REIT"]      = (0.55 * _ihsg
                          + np.sqrt(1 - 0.55**2) * _rng.normal(0, 0.008, _N))
    _base["Komoditas"] = (0.45 * _ihsg
                          + np.sqrt(1 - 0.45**2) * _rng.normal(0, 0.015, _N))
    df_returns = _base[ASSET_NAMES]
    LIVE_DATA  = _w1.LIVE_DATA

    _IHSG_CVAR_95   = _w1.risk_metrics["IHSG (Saham)"][0.95]["CVaR"]
    _IHSG_CVAR_99   = _w1.risk_metrics["IHSG (Saham)"][0.99]["CVaR"]
    _GARCH_PERSIST  = _w1.garch_results["IHSG (Saham)"]["params"]["persistence"]
    _GARCH_HALFLIFE = _w1.garch_results["IHSG (Saham)"]["params"]["half_life"]
    _GARCH_SERIES   = _w1.garch_results["IHSG (Saham)"]["annualized_vol"].tolist()

except Exception as _e:
    log.warning("dara_week1 gagal (%s) — fallback simulasi", _e)
    _N = 1260
    _MU = np.array([0.0003, 0.00015, 0.00008, 0.00020, 0.00025])
    _SG = np.array([0.012,  0.004,   0.0008,  0.008,   0.015])
    _CM = np.array([[1,.1,-.05,.55,.45],[.1,1,.2,-.1,-.15],[-.05,.2,1,-.08,-.1],
                    [.55,-.1,-.08,1,.3],[.45,-.15,-.1,.3,1]])
    _L  = np.linalg.cholesky(_CM)
    _sh = np.random.standard_t(df=5, size=(_N, 5)) / np.sqrt(5/3)
    df_returns = pd.DataFrame(
        _MU + (_sh @ _L.T) * _SG, columns=ASSET_NAMES,
        index=pd.bdate_range(end="2024-12-31", periods=_N))


# ─────────────────────────────────────────────────────────────────────────────
# SINYAL INPUT
# ─────────────────────────────────────────────────────────────────────────────
def compute_garch_vol(returns, alpha=0.10, beta=0.85):
    r     = returns[-500:] if len(returns) > 500 else returns
    var   = float(np.var(r))
    omega = var * (1 - alpha - beta)
    s2    = var
    for rt in r:
        s2 = omega + alpha * float(rt)**2 + beta * s2
    current = float(np.sqrt(max(s2, 1e-12))) * np.sqrt(252)
    p95     = float(np.percentile(
        [np.sqrt(max(omega + alpha * float(rt)**2 + beta * omega, 1e-12)) * np.sqrt(252)
         for rt in r], 95))
    return float(np.clip(current / (p95 + 1e-9), 0, 1))


def compute_momentum(df_ret, asset="IHSG (Saham)", lookback=60):
    recent = df_ret[asset].iloc[-lookback:].mean()
    hist   = df_ret[asset].mean()
    std    = df_ret[asset].std()
    return float((recent - hist) / (std + 1e-9))


def compute_correlation_regime(df_ret, window=60):
    corr  = df_ret.iloc[-window:].corr().values
    n     = len(corr)
    upper = corr[np.triu_indices(n, k=1)]
    return float(np.clip(np.mean(np.abs(upper)), 0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT BRIDGE
# ─────────────────────────────────────────────────────────────────────────────
class SentimentBridge:
    CACHE_FILE = "sentiment_output_signal.json"
    CACHE_TTL  = 1800

    def get(self, scenario="moderate"):
        """
        Priority chain:
          1. Cache JSON segar (< 30 menit) dari FinBERT atau demo_sentiment
          2. Live FinBERT (ProsusAI/finbert via sentiment_analysis.py)
          3. Live RSS + rule-based (demo_sentiment.py) — tanpa API key/torch
          4. Fallback stub per skenario (offline)
        """
        cached = self._load_cache()
        if cached:
            log.info("SentimentBridge: cache hit (source=%s)",
                     cached.get("source", "unknown"))
            return self._fmt(cached)

        # Layer 2: FinBERT penuh
        try:
            from sentiment_analysis import run_pipeline
            log.info("SentimentBridge: live FinBERT pipeline...")
            _, signal = run_pipeline()
            self._save_cache(signal)
            return self._fmt(signal)
        except Exception as exc:
            log.debug("FinBERT tidak tersedia (%s) — coba RSS fallback", exc)

        # Layer 3: RSS + rule-based (gratis, tanpa torch)
        try:
            from demo_sentiment import get_live_headlines_for_dara
            log.info("SentimentBridge: RSS rule-based sentiment...")
            result = get_live_headlines_for_dara(scenario)
            # Simpan ke cache supaya 30 menit berikutnya pakai cache
            if result.get("_signal"):
                self._save_cache(result["_signal"])
            return result
        except Exception as exc:
            log.debug("RSS sentiment gagal (%s) — pakai stub", exc)

        # Layer 4: stub
        log.info("SentimentBridge: stub fallback (scenario=%s)", scenario)
        return get_finbert_sentiment_stub(scenario)

    def refresh(self):
        """Paksa refresh: coba FinBERT dulu, lalu RSS."""
        # Coba FinBERT
        try:
            from sentiment_analysis import run_pipeline
            log.info("SentimentBridge.refresh: FinBERT pipeline...")
            _, signal = run_pipeline()
            self._save_cache(signal)
            return signal
        except Exception as exc:
            log.warning("FinBERT refresh gagal (%s) — coba RSS", exc)

        # Fallback ke RSS
        try:
            from demo_sentiment import run_demo_pipeline
            log.info("SentimentBridge.refresh: RSS fallback...")
            signal = run_demo_pipeline()
            self._save_cache(signal)
            return signal
        except Exception as exc:
            log.warning("RSS refresh juga gagal: %s", exc)
            return None

    def _save_cache(self, signal: dict):
        try:
            with open(self.CACHE_FILE, "w") as f:
                json.dump(signal, f, indent=2)
        except Exception as exc:
            log.debug("Cache save error: %s", exc)

    def _load_cache(self):
        if not os.path.exists(self.CACHE_FILE):
            return None
        if time.time() - os.path.getmtime(self.CACHE_FILE) > self.CACHE_TTL:
            return None
        try:
            with open(self.CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _fmt(sig):
        score = float(sig.get("avg_sentiment_score", 0.0))
        phase = sig.get("dominant_market_phase", "NEUTRAL")
        fg    = sig.get("fear_greed_index", 0.5)
        total = sig.get("n_articles_total", 0)
        pneg  = sig.get("pct_negative_news", 0)
        ppos  = sig.get("pct_positive_news", 0)
        if phase == "BEAR":
            hl = [f"Sentimen BEAR dari {total} artikel real-time",
                  f"Fear/Greed Index: {fg:.2f} (tekanan jual)",
                  f"Berita negatif: {pneg:.0%} dari total"]
        elif phase == "BULL":
            hl = [f"Sentimen BULL dari {total} artikel real-time",
                  f"Fear/Greed Index: {fg:.2f} (greed dominan)",
                  f"Berita positif: {ppos:.0%} dari total"]
        else:
            hl = [f"Sentimen NETRAL dari {total} artikel real-time",
                  f"Fear/Greed Index: {fg:.2f}",
                  f"Positif {ppos:.0%} | Negatif {pneg:.0%}"]
        return {"market": float(np.clip(score, -1, 1)), "headlines": hl, "_signal": sig}


_sentiment_bridge = SentimentBridge()


def get_finbert_sentiment_stub(scenario="moderate"):
    S = {
        "crisis":   {"market": -0.82, "headlines": [
            "Bank sentral darurat: suku bunga naik 150 bps sekaligus",
            "Rupiah sentuh level terendah, capital outflow Rp 45T dalam 3 hari",
            "IHSG turun 8% dalam seminggu, tekanan jual masih berlanjut"]},
        "bearish":  {"market": -0.42, "headlines": [
            "Inflasi melebihi ekspektasi, tekanan ke earning korporasi",
            "FII mencatat net sell Rp 8.2T di pasar saham",
            "PMI manufaktur kontraksi 3 bulan berturut-turut"]},
        "moderate": {"market":  0.05, "headlines": [
            "BI pertahankan suku bunga acuan di 6.25%",
            "GDP Q3 tumbuh 5.1%, sesuai konsensus",
            "Pasar menunggu sinyal Fed pada pertemuan FOMC"]},
        "bullish":  {"market":  0.48, "headlines": [
            "Fed memberi sinyal pivot, dovish tone mendominasi",
            "FDI masuk Rp 120T, rekor tertinggi sejak 2017",
            "IHSG tembus level 8.000, sentiment investor membaik"]},
        "euphoria": {"market":  0.78, "headlines": [
            "IHSG cetak all-time high 5 hari berturut-turut",
            "Crypto rally 40%, risk appetite tertinggi sejak 2021",
            "Leverage margin di broker naik 300% MoM"]},
    }
    return S.get(scenario, S["moderate"])


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: REGIME CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
class RiskRegimeClassifier:
    WEIGHTS = {"garch": 0.35, "sentiment": 0.30, "momentum": 0.20, "correlation": 0.15}
    GUARDRAILS = {
        "DEFENSIVE": {
            "IHSG (Saham)": (0.05, 0.20), "Obligasi (SBN)": (0.30, 0.55),
            "Pasar Uang":   (0.25, 0.45), "REIT":           (0.00, 0.10),
            "Komoditas":    (0.00, 0.10),
        },
        "MODERATE": {
            "IHSG (Saham)": (0.20, 0.45), "Obligasi (SBN)": (0.20, 0.40),
            "Pasar Uang":   (0.10, 0.25), "REIT":           (0.05, 0.20),
            "Komoditas":    (0.05, 0.15),
        },
        "AGGRESSIVE": {
            "IHSG (Saham)": (0.40, 0.65), "Obligasi (SBN)": (0.10, 0.25),
            "Pasar Uang":   (0.05, 0.15), "REIT":           (0.10, 0.25),
            "Komoditas":    (0.05, 0.20),
        },
    }
    REGIME_COLORS = {"DEFENSIVE": RED_RISK, "MODERATE": GOLD, "AGGRESSIVE": ACCENT_3}

    def classify(self, garch_norm, sentiment_mkt, momentum_z, corr_regime):
        scores = {
            "garch":       (1 - garch_norm) * 100,
            "sentiment":   (sentiment_mkt + 1) / 2 * 100,
            "momentum":    (float(np.clip(momentum_z, -3, 3)) + 3) / 6 * 100,
            "correlation": (1 - corr_regime) * 100,
        }
        total  = sum(scores[k] * self.WEIGHTS[k] for k in scores)
        regime = "DEFENSIVE" if total < 35 else "MODERATE" if total < 65 else "AGGRESSIVE"
        return {
            "regime":     regime,
            "score":      round(total, 2),
            "sub_scores": scores,
            "raw": {k: round(v, 3) for k, v in
                    zip(["garch","sentiment","momentum","correlation"],
                        [garch_norm, sentiment_mkt, momentum_z, corr_regime])},
            "guardrails": self.GUARDRAILS[regime],
            "color":      self.REGIME_COLORS[regime],
        }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: HRP OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
class HRPOptimizer:
    def _cluster_var(self, cov, items):
        sub = cov[np.ix_(items, items)]
        w   = np.ones(len(items)) / len(items)
        return float(w @ sub @ w)

    def _recursive_bisection(self, cov, items):
        w = pd.Series(1.0, index=items)
        clusters = [list(items)]
        while clusters:
            clusters = [sub[j:k] for sub in clusters
                        for j, k in ((0, len(sub)//2), (len(sub)//2, len(sub)))
                        if len(sub) > 1]
            for i in range(0, len(clusters) - 1, 2):
                Lc, Rc = clusters[i], clusters[i+1]
                vL, vR = self._cluster_var(cov, Lc), self._cluster_var(cov, Rc)
                alpha  = 1 - vL / (vL + vR + 1e-12)
                w[Lc] *= alpha
                w[Rc] *= (1 - alpha)
        return w

    def optimize(self, df_ret, guardrails):
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance  import squareform
        assets = df_ret.columns.tolist()
        cov    = df_ret.cov().values * 252
        std    = np.sqrt(np.diag(cov))
        corr   = np.clip(cov / np.outer(std, std), -1, 1)
        dist   = np.sqrt(np.clip(0.5 * (1 - corr), 0, 1))
        Z      = linkage(squareform(dist, checks=False), method="ward")
        order  = leaves_list(Z)
        raw_w  = self._recursive_bisection(cov, order)
        w      = pd.Series(raw_w.values, index=[assets[i] for i in order])[assets]
        for asset in assets:
            lo, hi = guardrails.get(asset, (0.0, 1.0))
            w[asset] = np.clip(w[asset], lo, hi)
        w /= w.sum()
        return {"weights": w, "cov": cov, "corr": corr,
                "linkage": Z, "sorted": [assets[i] for i in order], "assets": assets}


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE — dipanggil dara_server.py
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(scenario="moderate", use_live_sentiment=True):
    clf = RiskRegimeClassifier()
    opt = HRPOptimizer()

    garch_n = compute_garch_vol(df_returns["IHSG (Saham)"].values)
    mom_z   = compute_momentum(df_returns)
    corr_r  = compute_correlation_regime(df_returns)

    senti         = (_sentiment_bridge.get(scenario) if use_live_sentiment
                     else get_finbert_sentiment_stub(scenario))
    sentiment_mkt = float(senti["market"])
    headlines     = senti["headlines"]
    raw_signal    = senti.get("_signal", {})

    ri    = clf.classify(garch_n, sentiment_mkt, mom_z, corr_r)
    hr    = opt.optimize(df_returns, ri["guardrails"])
    w     = hr["weights"]
    w_vec = w.values
    cov   = hr["cov"]
    p_var = float(w_vec @ cov @ w_vec)
    p_vol = float(np.sqrt(p_var)) * 100

    rc = {a: round(w_vec[i] * (cov @ w_vec)[i] / (p_var + 1e-12) * 100, 2)
          for i, a in enumerate(hr["assets"])}

    # GARCH time-series untuk chart (252 hari terakhir)
    garch_series = (_GARCH_SERIES[-252:] if _GARCH_SERIES is not None
                    else [p_vol / np.sqrt(252)] * 252)

    return {
        "scenario":      scenario,
        "regime":        ri,
        "weights":       w.to_dict(),
        "portfolio_vol": round(p_vol, 2),
        "risk_contrib":  rc,
        "headlines":     headlines,
        "raw_inputs": {
            "garch_norm":  round(garch_n, 3),
            "sentiment":   round(sentiment_mkt, 3),
            "momentum_z":  round(mom_z, 3),
            "corr_regime": round(corr_r, 3),
        },
        "garch_params": {
            "alpha":       0.10,
            "beta":        0.85,
            "persistence": round(_GARCH_PERSIST, 3),
            "half_life":   round(_GARCH_HALFLIFE, 1),
        },
        "garch_series":  [round(v, 4) for v in garch_series],
        "cvar": {
            "ihsg_cvar95_pct": round(abs(_IHSG_CVAR_95) * 100, 3),
            "ihsg_cvar99_pct": round(abs(_IHSG_CVAR_99) * 100, 3),
        },
        "sentiment_signal": raw_signal,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DEMO  — python dara_week3.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DEMO = "bearish"
    print("=" * 65)
    print(f"  D.A.R.A — Minggu 3  |  Skenario: {DEMO.upper()}  |  LIVE: {LIVE_DATA}")
    print("=" * 65)

    res = run_pipeline(DEMO, use_live_sentiment=True)
    ri  = res["regime"]
    print(f"\n  REGIME : {ri['regime']}  |  SCORE : {ri['score']:.1f}/100")
    for k, v in ri["sub_scores"].items():
        bar = "█" * int(v / 5) + "░" * (20 - int(v / 5))
        print(f"  {k:<13}: [{bar}] {v:.1f}")

    print(f"\n  {'Aset':<22} {'Bobot':>7}  {'Risk Contrib':>13}")
    print("  " + "─" * 45)
    for a, wv in res["weights"].items():
        rc = res["risk_contrib"].get(a, 0)
        print(f"  {a:<22} {wv*100:>6.2f}%  {rc:>12.2f}%")
    print(f"\n  Portfolio Vol : {res['portfolio_vol']:.2f}%/thn")
    print(f"  IHSG CVaR 95% : -{res['cvar']['ihsg_cvar95_pct']:.3f}%/hari")
    print(f"  GARCH Persist : {res['garch_params']['persistence']}")

    clf2 = RiskRegimeClassifier(); opt2 = HRPOptimizer()
    gn = compute_garch_vol(df_returns["IHSG (Saham)"].values)
    mz = compute_momentum(df_returns); cr = compute_correlation_regime(df_returns)
    print(f"\n{'='*65}\n  SENSITIVITY TABLE\n{'='*65}")
    for sc in ["crisis","bearish","moderate","bullish","euphoria"]:
        s  = get_finbert_sentiment_stub(sc)
        r  = clf2.classify(gn, s["market"], mz, cr)
        hr = opt2.optimize(df_returns, r["guardrails"])
        ww = hr["weights"]
        mk = "  ◄" if sc == DEMO else ""
        print(f"  {sc:<10} {r['regime']:<12} {r['score']:>4.1f}  "
              f"IHSG:{ww['IHSG (Saham)']*100:>5.1f}%  "
              f"SBN:{ww['Obligasi (SBN)']*100:>5.1f}%  "
              f"PU:{ww['Pasar Uang']*100:>5.1f}%{mk}")
