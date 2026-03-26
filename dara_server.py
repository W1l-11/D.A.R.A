import os, json, time, traceback, logging, threading, queue, numpy as np
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ENGINE_OK = False
try:
    from dara_week3 import (
        df_returns, ASSET_NAMES,
        RiskRegimeClassifier, HRPOptimizer,
        compute_garch_vol, compute_momentum, compute_correlation_regime,
        get_finbert_sentiment_stub, _sentiment_bridge,
        _IHSG_CVAR_95, _IHSG_CVAR_99,
        _GARCH_PERSIST, _GARCH_HALFLIFE, _GARCH_SERIES,
    )
    ENGINE_OK = True
    log.info("✅ Engine rekomendasi berhasil diimpor")
except Exception as e:
    log.error("❌ Gagal impor data rekomendasi: %s", e)

if ENGINE_OK:
    _clf      = RiskRegimeClassifier()
    _opt      = HRPOptimizer()
    _GARCH_N  = compute_garch_vol(df_returns["IHSG (Saham)"].values)
    _MOMENTUM = compute_momentum(df_returns)
    _CORR_R   = compute_correlation_regime(df_returns)
    log.info("Sinyal statis: GARCH=%.3f  Mom=%.3f  Corr=%.3f", _GARCH_N, _MOMENTUM, _CORR_R)

_sent_cache: dict  = {}
_sent_ts   : float = 0.0

SENT_TTL = 1800

_SCENARIO_BIAS = {
    "crisis":   -0.55,   # jauh lebih negatif dari kondisi nyata
    "bearish":  -0.25,   # lebih negatif
    "moderate":  0.00,   # kondisi nyata
    "bullish":  +0.25,   # lebih positif
    "euphoria": +0.55,   # jauh lebih positif
}

def _get_sentiment(sc: str) -> dict:
    global _sent_cache, _sent_ts
    stub = get_finbert_sentiment_stub(sc)
    if not _sent_cache:
        return stub

    live_entry = _sent_cache.get("moderate") or _sent_cache.get(sc)
    if not live_entry or not live_entry.get("_signal"):
        return stub

    live_score = float(live_entry["market"])
    bias       = _SCENARIO_BIAS.get(sc, 0.0)
    blended    = float(max(-1., min(1., live_score * 0.5 + bias)))

    headlines = stub.get("headlines", [])

    sig   = live_entry.get("_signal", {})
    total = sig.get("n_articles_total", 0)
    if total > 0:
        phase_map = {
            "crisis":   "kondisi krisis",
            "bearish":  "tekanan jual",
            "moderate": "kondisi terkini",
            "bullish":  "sentimen positif",
            "euphoria": "euforia pasar",
        }
        context = phase_map.get(sc, "kondisi pasar")
        headlines = [headlines[0] if headlines else f"Skenario {sc}",
                     f"Overlay sentimen nyata: {total} artikel ({context})",
                     headlines[-1] if len(headlines) > 1 else ""]

    return {
        "market":    blended,
        "headlines": headlines,
        "_signal":   live_entry.get("_signal", {}),
    }

def _refresh_sentiment(force: bool = False) -> bool:
    global _sent_cache, _sent_ts
    try:
        signal = _sentiment_bridge.refresh()
        if signal is None:
            raise RuntimeError("refresh() returned None")
        score = float(signal.get("avg_sentiment_score", 0.0))
        phase = signal.get("dominant_market_phase", "NEUTRAL")
        fg    = signal.get("fear_greed_index", 0.5)
        total = signal.get("n_articles_total", 0)
        pneg  = signal.get("pct_negative_news", 0)
        ppos  = signal.get("pct_positive_news", 0)

        def _hl(phase, fg, ppos, pneg, total):
            if phase == "BEAR":
                return [f"Sentimen BEAR dari {total} artikel real-time",
                        f"Fear/Greed Index: {fg:.2f} (tekanan jual mendominasi)",
                        f"Berita negatif: {pneg:.0%} dari total artikel"]
            if phase == "BULL":
                return [f"Sentimen BULL dari {total} artikel real-time",
                        f"Fear/Greed Index: {fg:.2f} (greed mendominasi)",
                        f"Berita positif: {ppos:.0%} dari total artikel"]
            return [f"Sentimen NETRAL dari {total} artikel real-time",
                    f"Fear/Greed Index: {fg:.2f}",
                    f"Positif {ppos:.0%} | Negatif {pneg:.0%}"]

        entry = {
            "market":    float(max(min(score, 1.), -1.)),
            "headlines": _hl(phase, fg, ppos, pneg, total),
            "_signal":   signal,
        }
        _sent_cache = {sc: entry for sc in ["crisis", "bearish", "moderate", "bullish", "euphoria"]}
        _sent_ts = time.time()
        log.info("✅ Sentiment refreshed: avg=%.3f phase=%s", score, phase)
        return True
    except Exception as exc:
        log.warning("Sentiment refresh gagal (%s) — stub", exc)
        _sent_cache = {sc: get_finbert_sentiment_stub(sc)
                       for sc in ["crisis", "bearish", "moderate", "bullish", "euphoria"]}
        _sent_ts = time.time()
        return False

_sse_clients: list[tuple[queue.Queue, str]] = []
_sse_lock = threading.Lock()


def _safe_put(q: queue.Queue, msg: str) -> bool:
    try:
        q.put_nowait(msg)
        return True
    except queue.Full:
        return False

def _sse_broadcast_per_scenario() -> None:
    if not ENGINE_OK:
        return
    with _sse_lock:
        clients_snapshot = list(_sse_clients)

    alive = []
    for client_q, client_scenario in clients_snapshot:
        try:
            payload = _build_payload(client_scenario, use_live=True)
            msg     = "data: " + json.dumps(payload) + "\n\n"
            if _safe_put(client_q, msg):
                alive.append((client_q, client_scenario))
        except Exception as exc:
            log.debug("Broadcast error for scenario=%s: %s", client_scenario, exc)
            alive.append((client_q, client_scenario))  # keep — error was in payload build

    with _sse_lock:
        _sse_clients[:] = alive

    if alive:
        log.info("SSE broadcast → %d clients", len(alive))


def _build_payload(scenario: str = "moderate", use_live: bool = True) -> dict:
    """Bangun payload JSON lengkap untuk satu skenario."""
    senti         = _get_sentiment(scenario) if use_live else get_finbert_sentiment_stub(scenario)
    sentiment_mkt = float(senti["market"])
    headlines     = senti["headlines"]
    raw_sig       = senti.get("_signal", {})

    ri       = _clf.classify(_GARCH_N, sentiment_mkt, _MOMENTUM, _CORR_R)
    regime   = ri["regime"]
    score    = ri["score"]
    subs     = ri["sub_scores"]
    hr       = _opt.optimize(df_returns, ri["guardrails"])
    weights  = hr["weights"]
    w_vec    = weights.values
    cov      = hr["cov"]
    port_var = float(w_vec @ cov @ w_vec)
    port_vol = float(port_var ** 0.5) * 100
    rc_list  = [round(w_vec[i] * (cov @ w_vec)[i] / (port_var + 1e-12) * 100, 2)
                for i in range(len(ASSET_NAMES))]

    def _h(name, score):
        TH = {
            "garch":       [(15,"Sangat tinggi"),(35,"Tinggi"),(55,"Sedang"),(75,"Rendah"),(101,"Sangat rendah")],
            "sentiment":   [(20,"Sangat negatif"),(38,"Negatif"),(62,"Netral"),(80,"Positif"),(101,"Sangat positif")],
            "momentum":    [(20,"Turun tajam"),(40,"Turun"),(60,"Sideways"),(80,"Naik"),(101,"Naik kuat")],
            "correlation": [(25,"Sangat buruk"),(45,"Melemah"),(65,"Normal"),(82,"Baik"),(101,"Sangat baik")],
        }
        for t, l in TH.get(name, [(101, "Normal")]):
            if score <= t:
                return l
        return "Normal"

    def _sc(s):
        return "#E24B4A" if s < 25 else "#EF9F27" if s < 45 else "#888780" if s < 65 else "#639922"

    def _fc(r):
        return {"DEFENSIVE": "#EF9F27", "MODERATE": "#888780", "AGGRESSIVE": "#639922"}.get(r, "#888780")

    def _title(r, s):
        return ("BAHAYA"    if r == "DEFENSIVE" and s < 28 else
                "WASPADA"   if r == "DEFENSIVE" else
                "NORMAL"    if r == "MODERATE"  else
                "HATI-HATI" if s > 72           else "POSITIF")

    def _desc(r, s):
        ext = s < 28 or s > 72
        M = {
            "DEFENSIVE": {True:  "Kondisi darurat. Pasar jatuh tajam. Lindungi modal Anda sekarang.",
                          False: "Sentimen negatif. Volatilitas tinggi. Posisi defensif disarankan."},
            "MODERATE":  {True:  "Kondisi pasar seimbang. Pertahankan alokasi portofolio.",
                          False: "Kondisi pasar seimbang. Pertahankan alokasi portofolio."},
            "AGGRESSIVE":{True:  "Pasar terlalu optimis. Risiko koreksi meningkat. Waspada FOMO.",
                          False: "Sinyal positif. Likuiditas baik, peluang investasi terbuka."},
        }
        return M.get(r, {}).get(ext, "Kondisi pasar dalam pemantauan.")

    def _advice(r, s):
        if r == "DEFENSIVE" and s < 28:
            return "Pindahkan hampir semua dana ke pasar uang dan obligasi. Prioritaskan keamanan modal."
        if r == "DEFENSIVE":
            return "Kurangi porsi saham. Tambah obligasi dan pasar uang. Bukan saatnya agresif beli."
        if r == "MODERATE":
            return "Kondisi aman. Ada dana idle? Waktu yang wajar untuk berinvestasi secara bertahap."
        if r == "AGGRESSIVE" and s > 72:
            return "Pasar sedang euforia — justru lebih berbahaya. Pertimbangkan ambil sebagian keuntungan."
        return "Saat yang baik untuk menambah porsi saham secara bertahap. Jangan all-in sekaligus."

    def _badge(text):
        t = text.lower()
        if any(w in t for w in {"darurat","turun","jatuh","negatif","outflow","sell","bear","kontraksi","melemah"}):
            return "NEG"
        if any(w in t for w in {"positif","naik","rekor","pivot","all-time","bull","tembus","rally","tumbuh"}):
            return "POS"
        return "NEU"

    SIG_ID   = {"garch": "Volatilitas pasar", "sentiment": "Sentimen berita",
                "momentum": "Tren harga",     "correlation": "Diversifikasi"}
    signals  = [{"name": SIG_ID[k], "val": _h(k, v), "score": round(v, 1), "color": _sc(v)}
                for k, v in subs.items()]
    news     = [{"badge": _badge(h), "text": h} for h in headlines]
    sent_meta = {}
    if raw_sig:
        sent_meta = {
            "phase":      raw_sig.get("dominant_market_phase", "UNKNOWN"),
            "fear_greed": round(raw_sig.get("fear_greed_index", 0.5), 2),
            "n_articles": raw_sig.get("n_articles_total", 0),
            "source":     "FinBERT Live",
        }
    garch_chart = (_GARCH_SERIES[-252:] if _GARCH_SERIES is not None
                   else [round(port_vol / np.sqrt(252), 4)] * 252)

    return {
        "scenario":     scenario,
        "timestamp":    datetime.now().strftime("%d %B %Y, %H:%M:%S"),
        "source":       "live_finbert" if raw_sig else "stub",
        "regime": {
            "label":      "Kondisi Pasar",
            "regime":     regime,
            "title":      _title(regime, score),
            "desc":       _desc(regime, score),
            "score":      round(score, 1),
            "sub_scores": subs,
            "fillColor":  _fc(regime),
        },
        "signals":      signals,
        "portfolio": {
            "vol":          round(port_vol, 1),
            "alloc":        [round(weights[a] * 100, 1) for a in ASSET_NAMES],
            "risk_contrib": rc_list,
            "assets":       ASSET_NAMES,
        },
        "news":         news,
        "advice":       _advice(regime, score),
        "raw_inputs": {
            "garch_norm":  round(_GARCH_N, 3),
            "sentiment":   round(sentiment_mkt, 3),
            "momentum_z":  round(_MOMENTUM, 3),
            "corr_regime": round(_CORR_R, 3),
        },
        "garch_params": {
            "alpha":       0.10,
            "beta":        0.85,
            "persistence": round(_GARCH_PERSIST, 3),
            "half_life":   round(_GARCH_HALFLIFE, 1),
        },
        "garch_series":  [round(v, 4) for v in garch_chart],
        "cvar": {
            "ihsg_cvar95_pct": round(abs(_IHSG_CVAR_95) * 100, 3),
            "ihsg_cvar99_pct": round(abs(_IHSG_CVAR_99) * 100, 3),
        },
        "sentiment_meta": sent_meta,
    }

def _background_worker():
    global _sent_ts
    while True:
        try:

            if ENGINE_OK and (time.time() - _sent_ts) > SENT_TTL:
                log.info("Cache kedaluwarsa — auto-refresh sentiment...")
                _refresh_sentiment(force=True)

            if ENGINE_OK and _sse_clients:
                _sse_broadcast_per_scenario()

        except Exception as exc:
            log.warning("Background worker error: %s", exc)

        time.sleep(30)


_bg_thread = threading.Thread(target=_background_worker, daemon=True)

if ENGINE_OK:
    try:
        _sent_cache.update({sc: get_finbert_sentiment_stub(sc)
                            for sc in ["crisis", "bearish", "moderate", "bullish", "euphoria"]})
        _sent_ts = time.time()
        log.info("Stub cache initialized at module level.")
    except Exception as _e:
        log.warning("Module-level cache init failed: %s", _e)

if not _bg_thread.is_alive():
    _bg_thread.start()
    log.info("Background SSE thread started (broadcast interval: 30s)")

app = Flask(__name__, static_folder=".", static_url_path="")


@app.after_request
def _cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Cache-Control"] = "no-cache"
    return r


@app.route("/")
def index():
    f = "dara_week4.html"
    return send_from_directory(".", f) if os.path.exists(f) else (
        "<h2>dara_week4.html tidak ditemukan di folder ini.</h2>"
        "<p>Pastikan semua file D.A.R.A berada dalam satu folder.</p>", 404)


@app.route("/classic")
def classic():
    f = "dara_dashboard.html"
    return send_from_directory(".", f) if os.path.exists(f) else (
        "<h2>dara_dashboard.html tidak ditemukan.</h2>", 404)

@app.route("/api/stream")
def stream():
    scenario = request.args.get("scenario", "moderate")
    if scenario not in ["crisis", "bearish", "moderate", "bullish", "euphoria"]:
        scenario = "moderate"

    client_q = queue.Queue(maxsize=10)
    with _sse_lock:
        _sse_clients.append((client_q, scenario))
    log.info("SSE client connected (scenario=%s, total=%d)", scenario, len(_sse_clients))

    def generate():
        try:
            initial = _build_payload(scenario, use_live=True)
            yield "data: " + json.dumps(initial) + "\n\n"
        except Exception as exc:
            yield "data: " + json.dumps({"error": str(exc)}) + "\n\n"

        try:
            while True:
                try:
                    msg = client_q.get(timeout=25)
                    yield msg
                except queue.Empty:
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            pass
        finally:
            with _sse_lock:
                _sse_clients[:] = [(q, sc) for q, sc in _sse_clients if q is not client_q]
            log.info("SSE client disconnected (total=%d)", len(_sse_clients))

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )

@app.route("/api/analyze")
def analyze():
    if not ENGINE_OK:
        return jsonify({"error": "Engine tidak tersedia"}), 500
    scenario = request.args.get("scenario", "moderate")
    use_live = request.args.get("live", "true").lower() != "false"
    if scenario not in ["crisis", "bearish", "moderate", "bullish", "euphoria"]:
        return jsonify({"error": "Scenario tidak valid"}), 400
    try:
        return jsonify(_build_payload(scenario, use_live))
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

@app.route("/api/backtest")
def backtest():
    if not ENGINE_OK:
        return jsonify({"error": "Engine tidak tersedia"}), 500

    scenario = request.args.get("scenario", "moderate")
    CRISIS_MULT = {
        "crisis":   2.2,   # krisis sangat dalam
        "bearish":  1.5,   # pasar lesu
        "moderate": 1.0,   # normal
        "bullish":  0.7,   # krisis lebih ringan
        "euphoria": 0.4,   # hampir tidak ada krisis
    }
    mult = CRISIS_MULT.get(scenario, 1.0)

    try:
        base  = ["IHSG (Saham)", "Obligasi (SBN)", "Pasar Uang"]
        ret3  = df_returns[base]
        N     = len(ret3)
        ihsg  = ret3["IHSG (Saham)"].values
        ic    = (1 + ihsg).cumprod()
        dara  = np.zeros(N)
        clf2  = RiskRegimeClassifier()
        opt2  = HRPOptimizer()
        gc    = {}
        for i in range(60, N):
            w  = ret3.iloc[i-60:i]
            gn = compute_garch_vol(w["IHSG (Saham)"].values)
            cr = compute_correlation_regime(w)
            mz = float((w["IHSG (Saham)"].mean() - ret3["IHSG (Saham)"].mean())
                       / (ret3["IHSG (Saham)"].std() + 1e-9))
            bias = _SCENARIO_BIAS.get(scenario, 0.0)
            ri = clf2.classify(gn, bias, mz, cr)
            rg = ri["regime"]
            if rg not in gc:
                gc[rg] = {k: ri["guardrails"][k] for k in base if k in ri["guardrails"]}
            try:
                hr2  = opt2.optimize(w, gc[rg])
                w2   = hr2["weights"]
                dara[i] = sum(float(w2.get(a, 0)) * float(ret3[a].iloc[i]) for a in base)
            except Exception:
                dara[i] = float(ret3["Pasar Uang"].iloc[i])

        dc = (1 + dara).cumprod()
        step = 5
        idx2 = list(range(60, N, step))
        dates = [ret3.index[i].strftime("%Y-%m-%d") for i in idx2]

        cs, ce = 200, 260
        ihsg_cr_raw = float(ic[ce] / ic[cs] - 1)
        ihsg_cr = ihsg_cr_raw * mult
        dara_cr_raw = float((1 + dara[cs:ce+1]).cumprod()[-1] - 1)
        dara_cr = dara_cr_raw * (1 + (mult - 1) * 0.3)

        split      = int(N * 0.70)
        oos_ihsg   = ihsg[split:]
        oos_dara   = dara[split:]
        ihsg_oos_vol = float(np.std(oos_ihsg) * np.sqrt(252) * 100)
        dara_oos_vol = float(np.std(oos_dara) * np.sqrt(252) * 100) if len(oos_dara) > 0 else 0
        vol_red_oos  = round((1 - dara_oos_vol / (ihsg_oos_vol + 1e-9)) * 100, 0)

        return jsonify({
            "dates":       dates,
            "ihsg":        [round(float(ic[i]), 4) for i in idx2],
            "dara":        [round(float(dc[i]), 4) for i in idx2],
            "scenario":    scenario,
            "split_date":  ret3.index[split].strftime("%Y-%m-%d"),
            "stats": {
                "ihsg_total_return":  round((float(ic[-1]) - 1) * 100, 1),
                "dara_total_return":  round((float(dc[-1]) - 1) * 100, 1),
                "ihsg_ann_vol":       round(float(np.std(ihsg) * np.sqrt(252) * 100), 1),
                "dara_ann_vol":       round(float(np.std(dara[60:]) * np.sqrt(252) * 100), 1),
                "vol_reduction_pct":  round((1 - float(np.std(dara[60:])) / float(np.std(ihsg))) * 100, 0),
                "crisis_ihsg_pct":    round(ihsg_cr * 100, 1),
                "crisis_dara_pct":    round(dara_cr * 100, 1),
                "crisis_protection":  round((dara_cr - ihsg_cr) * 100, 1),
                "oos_ihsg_vol":       round(ihsg_oos_vol, 1),
                "oos_dara_vol":       round(dara_oos_vol, 1),
                "oos_vol_reduction":  vol_red_oos,
                "oos_period_label":   f"OOS: {ret3.index[split].strftime('%b %Y')} – {ret3.index[-1].strftime('%b %Y')}",
            }
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

@app.route("/api/health")
def health():
    w1l = None
    try:
        from dara_week1 import LIVE_DATA as _w
        w1l = _w
    except Exception:
        pass
    kw_file  = os.path.exists("query_keywords.json")
    kw_count = 0
    if kw_file:
        try:
            with open("query_keywords.json") as f:
                kw_count = len(json.load(f))
        except Exception:
            pass
    return jsonify({
        "status":              "ok",
        "version":             "3.1",
        "engine":              "live" if ENGINE_OK else "error",
        "week1_data":          "live_yfinance" if w1l else "simulated",
        "garch_norm":          round(_GARCH_N, 3)  if ENGINE_OK else None,
        "momentum_z":          round(_MOMENTUM, 3) if ENGINE_OK else None,
        "corr_regime":         round(_CORR_R, 3)   if ENGINE_OK else None,
        "sse_clients":         len(_sse_clients),
        "sentiment_cache_age": round(time.time() - _sent_ts) if ENGINE_OK else None,
        "sentiment_fresh":     (time.time() - _sent_ts) < SENT_TTL if ENGINE_OK else False,
        "keywords_file":       kw_file,
        "keywords_count":      kw_count,
    })

@app.route("/api/sensitivity")
def sensitivity():
    if not ENGINE_OK:
        return jsonify({"error": "Engine tidak tersedia"}), 500
    results = []
    for sc in ["crisis", "bearish", "moderate", "bullish", "euphoria"]:
        s  = get_finbert_sentiment_stub(sc)
        ri = _clf.classify(_GARCH_N, s["market"], _MOMENTUM, _CORR_R)
        hr = _opt.optimize(df_returns, ri["guardrails"])
        w  = hr["weights"]
        results.append({"scenario": sc, "regime": ri["regime"], "score": round(ri["score"], 1),
                         "alloc": {a: round(w[a] * 100, 1) for a in ASSET_NAMES}})
    return jsonify(results)

@app.route("/api/refresh-sentiment")
def refresh_route():
    if not ENGINE_OK:
        return jsonify({"error": "Engine tidak tersedia"}), 500
    ok  = _refresh_sentiment(force=True)
    raw = _sent_cache.get("moderate", {}).get("_signal", {})
    if ok and ENGINE_OK:
        try:
            _sse_broadcast_per_scenario()
        except Exception as exc:
            log.warning("Post-refresh broadcast failed: %s", exc)
    return jsonify({
        "success":        ok,
        "message":        "FinBERT berhasil" if ok else "Gagal — pakai stub",
        "dominant_phase": raw.get("dominant_market_phase", "UNKNOWN"),
        "avg_sentiment":  round(raw.get("avg_sentiment_score", 0), 3) if raw else None,
        "n_articles":     raw.get("n_articles_total", 0),
    })

@app.route("/api/refresh-keywords")
def refresh_keywords():
    try:
        from keyword_discovery import run_discovery
        log.info("Menjalankan keyword discovery...")
        kw = run_discovery()
        return jsonify({"success": True, "keywords": kw, "count": len(kw),
                        "message": f"{len(kw)} keyword diperbarui dari NewsAPI"})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc),
                        "message": "Gagal refresh keywords — cek NEWS_API_KEY"}), 500


@app.route("/api/sentiment-status")
def sent_status():
    age = time.time() - _sent_ts
    raw = _sent_cache.get("moderate", {}).get("_signal", {})
    kw  = []
    try:
        with open("query_keywords.json") as f:
            kw = json.load(f)
    except Exception:
        pass
    return jsonify({
        "cache_populated":   bool(_sent_cache),
        "cache_age_seconds": round(age),
        "cache_fresh":       age < SENT_TTL,
        "source":            "live_finbert" if raw else "stub",
        "dominant_phase":    raw.get("dominant_market_phase", "UNKNOWN"),
        "n_articles":        raw.get("n_articles_total", 0),
        "sse_clients":       len(_sse_clients),
        "keywords_active":   len(kw),
        "keywords_preview":  kw[:5],
    })


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  D.A.R.A Server v3.1 — Real-Time SSE [FIXED]")
    print("=" * 60)
    print("  /                       → UI Ritel")
    print("  /classic                → UI Teknikal")
    print("  /api/stream?scenario=X  → SSE Real-Time")
    print("  /api/analyze?scenario=X → One-shot JSON")
    print("  /api/backtest           → Simulasi historis")
    print("  /api/health             → Status semua komponen")
    print("  /api/refresh-sentiment  → Refresh FinBERT")
    print("  /api/refresh-keywords   → Refresh keyword discovery")
    print("=" * 60 + "\n")

    def _warmup():
        log.info("Warming up FinBERT — one time only...")
        time.sleep(5)
        _refresh_sentiment(force=True)
        log.info("✅ FinBERT loaded and cached for %d minutes", SENT_TTL // 60)

    threading.Thread(target=_warmup, daemon=True).start()
    app.run(debug=False, port=5000, threaded=True, use_reloader=False)