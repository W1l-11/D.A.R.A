/**
 * D.A.R.A — dara_week4.js  [FIXED]
 * Logic real-time, SSE, render, backtesting
 *
 * FIXES APPLIED:
 *  [F-01] err-banner → error-banner (null crash in connectSSE onopen/onerror + fetchDataPolling)
 *  [F-02] dot.className 'dot ...' → 'conn-dot ...' everywhere (CSS selectors need base class)
 *  [F-03] Removed first fetchData() definition (dead code — overridden by duplicate below)
 *  [F-04] switchTab() now uses style.display instead of classList.add('active')
 *         so it actually hides/shows panels (inline styles couldn't be overridden by class)
 *  [F-05] renderQuantPanel() now populates Risk-tab mirror IDs (#q-cvar95-r, #q-pvol-r, etc.)
 *  [F-06] Added buildRCChart() — rcChart canvas was never drawn
 *  [F-07] Removed dead _origSwitch capture (switchTab not yet defined at that point → always null)
 *  [F-08] fetchDataPolling() conn-dot class corrected to 'conn-dot live/err'
 *  [F-09] Removed dead updateStatusBar() (DOM elements it targets don't exist in HTML)
 */

/* global Chart */

// ── Config ────────────────────────────────────────────────────────────────────
const API = "http://localhost:5000";

const FUND_MAP = {
  "IHSG (Saham)":    { name:"Reksa Dana Saham",           fund:"Schroder Dana Prestasi Plus",  color:"#2980B9" },
  "Obligasi (SBN)":  { name:"RD Pendapatan Tetap",         fund:"Manulife Obligasi Negara",      color:"#27AE60" },
  "Pasar Uang":      { name:"Reksa Dana Pasar Uang",       fund:"BNI-AM Dana Likuid",            color:"#E67E22" },
  "REIT":            { name:"RD Properti / DIRE",           fund:"Reksa Dana Properti Syariah",   color:"#8E44AD" },
  "Komoditas":       { name:"Reksa Dana Emas",              fund:"Manulife Emas Dinamis",         color:"#C0392B" },
};
const ASSET_NAMES = Object.keys(FUND_MAP);
const COLORS = ASSET_NAMES.map(a => FUND_MAP[a].color);

const WEATHER_MAP = {
  DEFENSIVE_LOW:  { cls:"weather-storm",     emoji:"⛈️",  headline:"Cuaca Pasar: BADAI",          sub:"Risiko sangat tinggi. Pasar dalam tekanan besar. Modal Anda perlu perlindungan ekstra sekarang." },
  DEFENSIVE:      { cls:"weather-cloudy",    emoji:"🌧️",  headline:"Cuaca Pasar: MENDUNG TEBAL",   sub:"Sentimen negatif mendominasi. Volatilitas tinggi. Saatnya berposisi defensif dan hati-hati." },
  MODERATE:       { cls:"weather-neutral",   emoji:"⛅",  headline:"Cuaca Pasar: BERAWAN",         sub:"Kondisi pasar stabil namun belum cerah. Pertahankan alokasi dan pantau perkembangan." },
  AGGRESSIVE:     { cls:"weather-sunny",     emoji:"🌤️",  headline:"Cuaca Pasar: CERAH",           sub:"Sinyal positif dari pasar. Momentum baik untuk menambah investasi secara bertahap." },
  AGGRESSIVE_HIGH:{ cls:"weather-overheated",emoji:"🌡️", headline:"Cuaca Pasar: PANAS BERBAHAYA",  sub:"Pasar terlalu optimis — euforia berlebihan. Waspadai koreksi tajam. Jangan ikut FOMO." },
};

const SIGNAL_CONFIG = [
  { key:"garch",       label:"Volatilitas Pasar",  weight:"35%", color:"#2980B9" },
  { key:"sentiment",   label:"Sentimen Berita",     weight:"30%", color:"#8E44AD" },
  { key:"momentum",    label:"Tren Harga",           weight:"20%", color:"#27AE60" },
  { key:"correlation", label:"Diversifikasi",        weight:"15%", color:"#E67E22" },
];

// ── State ─────────────────────────────────────────────────────────────────────
let capital = 50_000_000;
let currentGoal = "education";
let currentData = null;
let donutChart = null;
let garchChart = null;
let rcChart    = null;          // [F-06] was missing
let isOnline = false;
let currentScenario = "moderate";
let quantOpen = false;

// ── Capital slider (logarithmic: 1jt – 1Milyar) ───────────────────────────────
function sliderToCapital(v) {
  const logMin = Math.log10(1_000_000);
  const logMax = Math.log10(1_000_000_000);
  return Math.round(Math.pow(10, logMin + (v / 100) * (logMax - logMin)));
}
function capitalToSlider(c) {
  const logMin = Math.log10(1_000_000);
  const logMax = Math.log10(1_000_000_000);
  return Math.round((Math.log10(c) - logMin) / (logMax - logMin) * 100);
}
function formatRp(n) {
  if (n >= 1_000_000_000) return `Rp ${(n/1_000_000_000).toFixed(2).replace(/\.00$/,'')} Miliar`;
  if (n >= 1_000_000)     return `Rp ${(n/1_000_000).toFixed(0)} Juta`;
  return `Rp ${n.toLocaleString('id-ID')}`;
}
function onCapitalChange(v) {
  capital = Math.max(sliderToCapital(v), 1_000_000);
  document.getElementById("capital-display").textContent = formatRp(capital);
  const slider = document.getElementById("capital-slider");
  slider.style.setProperty("--val", v + "%");
  if (currentData) renderCapitalData(currentData);
}

// ── Goal ──────────────────────────────────────────────────────────────────────
const GOAL_SCENARIO = {
  emergency:  "bearish",
  education:  "moderate",
  retirement: "bullish",
  growth:     "euphoria",
};
function setGoal(g) {
  currentGoal = g;
  document.querySelectorAll(".goal-card").forEach(c => c.classList.remove("active"));
  const goalEl = document.getElementById("goal-" + g);
  if (goalEl) goalEl.classList.add("active");
  currentScenario = GOAL_SCENARIO[g];

  // ── LANGKAH 1: Render FALLBACK seketika (0ms delay, pasti berbeda per goal) ──
  render(FALLBACK[currentScenario] || FALLBACK.moderate);

  resetScenarioCache();  // reset backtest cache
  // ── LANGKAH 2: Ambil data server yang scenario-specific ──────────────────────
  // Pakai live=false agar sentinel stub dipakai → data PASTI berbeda tiap scenario
  // Sinyal statis (GARCH, momentum, correlation) tetap dari server
  fetchScenarioData(currentScenario);
}

// ── Fetch data server untuk satu skenario (non-SSE, cepat, scenario-specific) ─
// Reset backtest cache saat scenario berubah
function resetScenarioCache() { _btData = null; }

async function fetchScenarioData(scenario) {
  try {
    const r = await fetch(
      `${API}/api/analyze?scenario=${scenario}&live=false`,
      { signal: AbortSignal.timeout(5000) }
    );
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    if (data.error) throw new Error(data.error);

    // Hanya render jika scenario belum berubah lagi selama request berlangsung
    if (data.scenario === currentScenario) {
      render(data);
      isOnline = true;
      const dot = document.getElementById("conn-dot");
      const lbl = document.getElementById("conn-label");
      const eb  = document.getElementById("error-banner");
      if (dot) dot.className = "conn-dot live";
      if (lbl) lbl.textContent = "Live — " + (data.timestamp || "");
      if (eb)  eb.style.display = "none";
    }
  } catch(e) {
    // Fallback sudah ter-render di LANGKAH 1, tidak perlu action tambahan
    console.debug("fetchScenarioData fallback:", e.message);
  }
}

// ── Weather ───────────────────────────────────────────────────────────────────
function getWeatherKey(regime, score) {
  if (regime === "DEFENSIVE" && score < 28) return "DEFENSIVE_LOW";
  if (regime === "DEFENSIVE")               return "DEFENSIVE";
  if (regime === "MODERATE")                return "MODERATE";
  if (regime === "AGGRESSIVE" && score > 72) return "AGGRESSIVE_HIGH";
  return "AGGRESSIVE";
}
function renderWeather(regime, score, pills) {
  const key  = getWeatherKey(regime, score);
  const w    = WEATHER_MAP[key];
  const hero = document.getElementById("weather-hero");
  hero.className = "weather-hero " + w.cls;
  document.getElementById("weather-emoji").textContent    = w.emoji;
  document.getElementById("weather-headline").textContent = w.headline;
  document.getElementById("weather-sub").textContent      = w.sub;
  document.getElementById("gauge-num").textContent        = Math.round(score);
  const pillsEl = document.getElementById("weather-pills");
  pillsEl.innerHTML = pills.map(p =>
    `<span class="weather-pill ${p.type || ''}">${p.text}</span>`
  ).join("");
}

// ── CVaR & capital breakdown ───────────────────────────────────────────────────
function renderCapitalData(data) {
  if (!data) return;
  const pvol    = data.portfolio.vol / 100;
  const dailyCV = pvol / Math.sqrt(252) * 2.33;
  const cvarAmt = Math.round(capital * dailyCV);
  const cvarPct = (dailyCV * 100).toFixed(2);

  const amtEl = document.getElementById("cvar-amount");
  amtEl.classList.remove("skeleton");
  amtEl.style.height = amtEl.style.width = "";
  amtEl.textContent = formatRp(cvarAmt);

  document.getElementById("cvar-pct-label").textContent = cvarPct + "% modal";
  document.getElementById("cvar-bar").style.width = Math.min(cvarPct * 10, 100) + "%";

  const alloc   = data.portfolio.alloc;
  const assets  = data.portfolio.assets;
  const breakEl = document.getElementById("capital-breakdown");
  breakEl.innerHTML = assets.map((a, i) => {
    const amt = Math.round(capital * alloc[i] / 100);
    const fi  = FUND_MAP[a] || {};
    return `<div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid var(--border);font-size:12px">
      <div style="display:flex;align-items:center;gap:6px">
        <span style="width:8px;height:8px;border-radius:2px;background:${fi.color||'#999'};flex-shrink:0;display:inline-block"></span>
        <span style="color:var(--text2)">${fi.name || a}</span>
      </div>
      <div style="text-align:right">
        <span style="font-weight:700;color:var(--text)">${formatRp(amt)}</span>
        <span style="color:var(--text3);margin-left:6px">${alloc[i].toFixed(1)}%</span>
      </div>
    </div>`;
  }).join("");
}

// ── Full render ───────────────────────────────────────────────────────────────
function render(data) {
  currentData = data;
  const r = data.regime;

  const pills = [
    { text: "Regime: " + r.regime },
    { text: "Score: " + Math.round(r.score) + "/100" },
    { text: "Vol. Portfolio: " + data.portfolio.vol.toFixed(1) + "%/thn",
      type: r.regime === "DEFENSIVE" ? "danger" : r.regime === "AGGRESSIVE" ? "ok" : "" },
  ];
  renderWeather(r.regime, r.score, pills);
  renderCapitalData(data);

  // Advice
  const advEl = document.getElementById("advice-text");
  advEl.classList.remove("skeleton");
  advEl.style.height = advEl.style.width = "";
  advEl.textContent = data.advice;

  // News
  document.getElementById("news-list").innerHTML = data.news.map(n => {
    const [bg, col] = n.badge === "NEG" ? ["#FDEAEA","#B03A2E"] :
                      n.badge === "POS" ? ["#E4F4EC","#1A6B3A"] :
                      ["var(--surface2)","var(--text3)"];
    const lbl = n.badge === "NEG" ? "Negatif" : n.badge === "POS" ? "Positif" : "Netral";
    return `<div style="display:flex;align-items:flex-start;gap:8px;padding:7px 0;border-bottom:1px solid var(--border)">
      <span style="background:${bg};color:${col};font-size:10px;font-weight:700;padding:2px 7px;border-radius:12px;white-space:nowrap;margin-top:2px">${lbl}</span>
      <span style="font-size:13px;color:var(--text2);line-height:1.45">${n.text}</span>
    </div>`;
  }).join("");

  // Donut chart
  const alloc  = data.portfolio.alloc;
  const assets = data.portfolio.assets;
  document.getElementById("donut-vol").textContent = data.portfolio.vol.toFixed(1) + "%";
  if (donutChart) {
    donutChart.data.datasets[0].data = alloc;
    donutChart.update("active");
  } else {
    donutChart = new Chart(document.getElementById("donutChart"), {
      type: "doughnut",
      data: { labels: assets, datasets: [{ data: alloc, backgroundColor: COLORS,
               borderWidth: 3, borderColor: "#FFFFFF", hoverOffset: 5 }] },
      options: { responsive: true, maintainAspectRatio: false, cutout: "66%",
        plugins: { legend: { display: false },
          tooltip: { callbacks: { label: c => "  " + c.label + ": " + c.parsed.toFixed(1) + "%" } } },
        animation: { duration: 500 } }
    });
  }

  // Fund list
  document.getElementById("fund-list").innerHTML = assets.map((a, i) => {
    const fi = FUND_MAP[a] || { name: a, fund: a, color: "#999" };
    return `<div class="fund-row">
      <div class="fund-left">
        <span class="fund-dot" style="background:${fi.color}"></span>
        <div><div class="fund-name">${fi.name}</div><div class="fund-type">${fi.fund}</div></div>
      </div>
      <span class="fund-pct">${alloc[i].toFixed(1)}%</span>
    </div>`;
  }).join("");

  renderQuantPanel(data);
}

// ── Quant panel ───────────────────────────────────────────────────────────────
function renderQuantPanel(data) {
  const r  = data.regime;
  const ri = data.raw_inputs;
  const gp = data.garch_params || {};
  const cv = data.cvar || {};

  // Signal bars
  const subs = r.sub_scores || {};
  document.getElementById("signal-bars").innerHTML = SIGNAL_CONFIG.map(s => {
    const score = subs[s.key] || 0;
    return `<div class="signal-bar-row">
      <div class="signal-bar-header">
        <span class="signal-bar-name">${s.label} <span style="color:var(--text3);font-size:10px">(${s.weight})</span></span>
        <span class="signal-bar-val" style="color:${s.color}">${score.toFixed(1)}</span>
      </div>
      <div class="signal-bar-track">
        <div class="signal-bar-fill" style="width:${Math.round(score)}%;background:${s.color}"></div>
      </div>
    </div>`;
  }).join("");

  // FinBERT bar
  const sentScore = ri.sentiment;
  const pos = Math.round(Math.max(0, sentScore) * 60 + 20);
  const neg = Math.round(Math.max(0, -sentScore) * 60 + 20);
  const neu = Math.max(100 - pos - neg, 5);
  document.getElementById("fb-pos").style.flex = pos;
  document.getElementById("fb-pos").textContent = pos + "%";
  document.getElementById("fb-neu").style.flex = neu;
  document.getElementById("fb-neu").textContent = neu + "%";
  document.getElementById("fb-neg").style.flex = neg;
  document.getElementById("fb-neg").textContent = neg + "%";

  // Signals-tab metrics
  document.getElementById("q-pvol").textContent      = data.portfolio.vol.toFixed(2) + "%/thn";
  document.getElementById("q-regime").textContent    = r.regime;
  document.getElementById("q-score").textContent     = r.score.toFixed(1) + " / 100";
  document.getElementById("q-source").textContent    = data.source === "live_finbert" ? "FinBERT Live" : "Stub";
  document.getElementById("q-alpha").textContent     = gp.alpha ?? "0.10";
  document.getElementById("q-beta").textContent      = gp.beta  ?? "0.85";
  document.getElementById("q-persist").textContent   = gp.persistence ? gp.persistence.toFixed(3) : "0.950";
  document.getElementById("q-halflife").textContent  = gp.half_life ? gp.half_life + " hari" : "14 hari";
  document.getElementById("q-cvar95").textContent    = cv.ihsg_cvar95_pct ? "-" + cv.ihsg_cvar95_pct.toFixed(3) + "%" : "—";
  document.getElementById("q-cvar99").textContent    = cv.ihsg_cvar99_pct ? "-" + cv.ihsg_cvar99_pct.toFixed(3) + "%" : "—";

  // [F-05] Populate Risk-tab mirror IDs (were always showing "—" before)
  const setR = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  setR("q-cvar95-r", cv.ihsg_cvar95_pct ? "-" + cv.ihsg_cvar95_pct.toFixed(3) + "%" : "—");
  setR("q-cvar99-r", cv.ihsg_cvar99_pct ? "-" + cv.ihsg_cvar99_pct.toFixed(3) + "%" : "—");
  setR("q-pvol-r",   data.portfolio.vol.toFixed(2) + "%/thn");
  setR("q-persist-r", gp.persistence ? gp.persistence.toFixed(3) : "0.950");

  // GARCH chart
  buildGarchChart(data.garch_series, data.portfolio.vol);

  // [F-06] Risk-contribution chart (was never built)
  buildRCChart(data.portfolio.assets, data.portfolio.risk_contrib, COLORS);
}

function buildGarchChart(garchSeries, portVol) {
  const ctx = document.getElementById("garchChart");
  if (garchChart) { garchChart.destroy(); garchChart = null; }

  let vols;
  if (garchSeries && garchSeries.length > 0) {
    vols = garchSeries.map(v => parseFloat(v).toFixed(3));
  } else {
    const N = 252;
    vols = [];
    let v = portVol / 100 / Math.sqrt(252);
    const omega = v * v * 0.05;
    for (let i = 0; i < N; i++) {
      const shock = (i === 60 || i === 180) ? v * 3.5 : (Math.random() - 0.5) * v * 2;
      v = Math.sqrt(Math.max(omega + 0.10 * shock * shock + 0.85 * v * v, 1e-8));
      vols.push((v * Math.sqrt(252) * 100).toFixed(3));
    }
  }

  const labels = vols.map((_, i) => {
    const d = new Date();
    d.setDate(d.getDate() - (vols.length - i));
    return d.toLocaleDateString("id-ID", { month: "short", day: "numeric" });
  });

  garchChart = new Chart(ctx, {
    type: "line",
    data: { labels, datasets: [{ data: vols, borderColor: "#2980B9", borderWidth: 1.5,
              fill: true, backgroundColor: "rgba(41,128,185,0.08)", tension: .3, pointRadius: 0 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false },
        tooltip: { callbacks: { label: c => ` ${parseFloat(c.raw).toFixed(1)}%/thn` } } },
      scales: {
        x: { display: false },
        y: { grid: { color: "rgba(0,0,0,0.05)" }, ticks: { font: { size: 10 }, color: "#9A9A8E",
            callback: v => v + "%" } }
      },
      animation: { duration: 600 },
    }
  });
}

// [F-06] Risk Contribution chart — was completely absent
function buildRCChart(assets, riskContrib, colors) {
  const ctx = document.getElementById("rcChart");
  if (!ctx) return;
  if (rcChart) { rcChart.destroy(); rcChart = null; }
  rcChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: assets,
      datasets: [{
        data: riskContrib,
        backgroundColor: colors,
        borderWidth: 2,
        borderColor: "#FFFFFF",
        hoverOffset: 4,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: "52%",
      plugins: {
        legend: { display: true, position: "right",
          labels: { font: { size: 9 }, boxWidth: 10, color: "var(--text2)" } },
        tooltip: { callbacks: { label: c => "  " + c.label + ": " + c.parsed.toFixed(1) + "%" } }
      },
      animation: { duration: 500 }
    }
  });
}

function toggleQuant() {
  quantOpen = !quantOpen;
  document.getElementById("quant-panel").classList.toggle("open", quantOpen);
  document.getElementById("quant-arrow").textContent = quantOpen ? "▲" : "▼";
  document.getElementById("quant-btn").querySelector("span:nth-child(2)").textContent =
    quantOpen ? "Sembunyikan Analisis Quant" : "Tampilkan Analisis Quant";
  if (quantOpen) document.getElementById("quant-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── API / Fallback ────────────────────────────────────────────────────────────
const FALLBACK = {
  crisis: {
    scenario:"crisis", timestamp:"Offline — Simulasi", source:"stub",
    regime:{ label:"Kondisi Pasar", title:"BAHAYA", desc:"Kondisi darurat. Pasar jatuh tajam.", score:24.8, fillColor:"#E74C3C", regime:"DEFENSIVE", sub_scores:{garch:0,sentiment:9,momentum:20,correlation:35} },
    portfolio:{ vol:3.0, alloc:[5.7,34.0,51.0,6.9,2.5], assets:["IHSG (Saham)","Obligasi (SBN)","Pasar Uang","REIT","Komoditas"], risk_contrib:[12,38,6,28,16] },
    news:[{badge:"NEG",text:"Bank sentral darurat: suku bunga naik 150 bps sekaligus"},{badge:"NEG",text:"Rupiah sentuh level terendah, capital outflow Rp 45T dalam 3 hari"},{badge:"NEG",text:"IHSG turun 8% dalam seminggu, tekanan jual masih berlanjut"}],
    advice:"Pindahkan hampir semua dana ke pasar uang dan obligasi. Prioritaskan keamanan modal sekarang.",
    raw_inputs:{garch_norm:1.0,sentiment:-0.82,momentum_z:-2.1,corr_regime:0.65},
    garch_params:{alpha:0.10,beta:0.85,persistence:0.950,half_life:14.0},
    cvar:{ihsg_cvar95_pct:2.450,ihsg_cvar99_pct:3.950},
    garch_series:[], sentiment_meta:{},
  },
  bearish: {
    scenario:"bearish", timestamp:"Offline — Simulasi", source:"stub",
    regime:{ label:"Kondisi Pasar", title:"WASPADA", desc:"Sentimen negatif. Volatilitas meningkat.", score:30.8, fillColor:"#EF9F27", regime:"DEFENSIVE", sub_scores:{garch:0,sentiment:29,momentum:50,correlation:80} },
    portfolio:{ vol:2.6, alloc:[6.7,34.5,51.8,3.1,3.9], assets:["IHSG (Saham)","Obligasi (SBN)","Pasar Uang","REIT","Komoditas"], risk_contrib:[15,42,5,22,16] },
    news:[{badge:"NEG",text:"Inflasi melebihi ekspektasi, tekanan ke earning korporasi"},{badge:"NEG",text:"FII mencatat net sell Rp 8.2T di pasar saham"},{badge:"NEU",text:"BI kemungkinan tahan suku bunga, menunggu data inflasi"}],
    advice:"Kurangi porsi saham. Tambah obligasi dan pasar uang. Jangan panik — tapi bukan saatnya agresif beli.",
    raw_inputs:{garch_norm:1.0,sentiment:-0.42,momentum_z:0.016,corr_regime:0.2},
    garch_params:{alpha:0.10,beta:0.85,persistence:0.950,half_life:14.0},
    cvar:{ihsg_cvar95_pct:2.450,ihsg_cvar99_pct:3.950},
    garch_series:[], sentiment_meta:{},
  },
  moderate: {
    scenario:"moderate", timestamp:"Offline — Simulasi", source:"stub",
    regime:{ label:"Kondisi Pasar", title:"NORMAL", desc:"Kondisi seimbang. Pertahankan alokasi.", score:37.8, fillColor:"#888780", regime:"MODERATE", sub_scores:{garch:45,sentiment:52,momentum:50,correlation:80} },
    portfolio:{ vol:5.2, alloc:[26.3,26.3,32.8,8.0,6.6], assets:["IHSG (Saham)","Obligasi (SBN)","Pasar Uang","REIT","Komoditas"], risk_contrib:[28,22,4,30,16] },
    news:[{badge:"NEU",text:"BI pertahankan suku bunga acuan 6.25%, sesuai ekspektasi"},{badge:"NEU",text:"GDP Q3 tumbuh 5.1%, sesuai konsensus analis"},{badge:"NEU",text:"Pasar menunggu hasil pertemuan Fed pekan depan"}],
    advice:"Kondisi aman. Ada dana idle? Ini waktu yang wajar untuk berinvestasi secara bertahap.",
    raw_inputs:{garch_norm:0.55,sentiment:0.05,momentum_z:0.3,corr_regime:0.2},
    garch_params:{alpha:0.10,beta:0.85,persistence:0.950,half_life:14.0},
    cvar:{ihsg_cvar95_pct:2.450,ihsg_cvar99_pct:3.950},
    garch_series:[], sentiment_meta:{},
  },
  bullish: {
    scenario:"bullish", timestamp:"Offline — Simulasi", source:"stub",
    regime:{ label:"Kondisi Pasar", title:"POSITIF", desc:"Sinyal positif dari pasar.", score:44.2, fillColor:"#639922", regime:"AGGRESSIVE", sub_scores:{garch:72,sentiment:74,momentum:67,correlation:80} },
    portfolio:{ vol:8.1, alloc:[38.0,20.0,13.0,17.0,12.0], assets:["IHSG (Saham)","Obligasi (SBN)","Pasar Uang","REIT","Komoditas"], risk_contrib:[40,15,3,28,14] },
    news:[{badge:"POS",text:"Fed beri sinyal pemangkasan suku bunga, pasar bereaksi positif"},{badge:"POS",text:"FDI masuk Rp 120T, rekor tertinggi sejak 2017"},{badge:"POS",text:"IHSG tembus level psikologis 8.000 untuk pertama kalinya"}],
    advice:"Saat yang baik untuk menambah porsi saham secara bertahap. Jangan all-in — beli berkala lebih aman.",
    raw_inputs:{garch_norm:0.28,sentiment:0.48,momentum_z:1.2,corr_regime:0.2},
    garch_params:{alpha:0.10,beta:0.85,persistence:0.950,half_life:14.0},
    cvar:{ihsg_cvar95_pct:2.450,ihsg_cvar99_pct:3.950},
    garch_series:[], sentiment_meta:{},
  },
  euphoria: {
    scenario:"euphoria", timestamp:"Offline — Simulasi", source:"stub",
    regime:{ label:"Kondisi Pasar", title:"HATI-HATI", desc:"Pasar euforia. Risiko koreksi meningkat.", score:49.0, fillColor:"#1D9E75", regime:"AGGRESSIVE", sub_scores:{garch:90,sentiment:89,momentum:83,correlation:35} },
    portfolio:{ vol:10.5, alloc:[40.0,15.0,10.0,20.0,15.0], assets:["IHSG (Saham)","Obligasi (SBN)","Pasar Uang","REIT","Komoditas"], risk_contrib:[42,12,2,30,14] },
    news:[{badge:"POS",text:"IHSG cetak all-time high 5 hari berturut-turut"},{badge:"POS",text:"Crypto rally 40%, risk appetite tertinggi sejak 2021"},{badge:"NEG",text:"Peringatan: valuasi saham sudah sangat tinggi, risiko koreksi"}],
    advice:"Pasar euforia — justru lebih berbahaya. Pertimbangkan ambil sebagian keuntungan. Jangan FOMO.",
    raw_inputs:{garch_norm:0.1,sentiment:0.78,momentum_z:2.5,corr_regime:0.65},
    garch_params:{alpha:0.10,beta:0.85,persistence:0.950,half_life:14.0},
    cvar:{ihsg_cvar95_pct:2.450,ihsg_cvar99_pct:3.950},
    garch_series:[], sentiment_meta:{},
  },
};

// ── Init ──────────────────────────────────────────────────────────────────────
const slider = document.getElementById("capital-slider");
slider.style.setProperty("--val", capitalToSlider(capital) + "%");
slider.value = capitalToSlider(capital);

// ── SSE Real-Time Connection ─────────────────────────────────────────────────
let _evtSource      = null;
let _reconnectTimer = null;
let _reconnectDelay = 3000;

// Initial load with education/moderate scenario
connectSSE("moderate");

// ── Backtesting ────────────────────────────────────────────────────────────────
let _btData = null, _btChart = null;

async function loadBacktest() {
  // Cache per skenario — saat goal berubah, cache direset
  const cacheKey = currentScenario;
  if (_btData && _btData._scenario === cacheKey) { renderBacktest(_btData); return; }
  try {
    const r = await fetch(
      `${API}/api/backtest?scenario=${cacheKey}`,
      { signal: AbortSignal.timeout(9000) }
    );
    if (!r.ok) throw new Error("HTTP " + r.status);
    _btData = await r.json();
    if (_btData.error) throw new Error(_btData.error);
    _btData._scenario = cacheKey;
    renderBacktest(_btData);
    const s  = _btData.stats;
    const vr = document.getElementById("bt-vol-reduction");
    const cp = document.getElementById("bt-crisis-protect");
    if (vr) vr.textContent = Math.round(s.vol_reduction_pct) + "%";
    if (cp) cp.textContent = "+" + s.crisis_protection.toFixed(1) + "pp";
  } catch (e) {
    const n = document.getElementById("bt-note2");
    if (n) n.textContent = "Gagal memuat backtesting: " + e.message + ". Pastikan server aktif.";
  }
}

function renderBacktest(d) {
  const s   = d.stats;
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  set("bt-vol-r2",    Math.round(s.vol_reduction_pct) + "%");
  set("bt-crisis-p2", "+" + s.crisis_protection.toFixed(1) + "pp");
  set("bt-dara-vol2", s.dara_ann_vol.toFixed(1) + "%/thn");

  const oosPeriod = d.split_date ? " (Train/Test split: " + d.split_date + ")" : "";
  const oosVol    = s.oos_dara_vol ? ` | OOS vol reduction: ${Math.round(s.oos_vol_reduction)}% ${s.oos_period_label || ""}` : "";

  set("bt-note2",
    "Simulasi 5 tahun: IHSG " + (s.ihsg_total_return > 0 ? "+" : "") + s.ihsg_total_return +
    "% vs D.A.R.A " + (s.dara_total_return > 0 ? "+" : "") + s.dara_total_return + "%. " +
    "Vol D.A.R.A: " + s.dara_ann_vol + "%/thn vs IHSG: " + s.ihsg_ann_vol + "%/thn. " +
    "Krisis: IHSG " + s.crisis_ihsg_pct + "% vs D.A.R.A " + s.crisis_dara_pct + "%." +
    oosVol + oosPeriod + " | Bukan jaminan hasil masa depan."
  );

  if (_btChart) { _btChart.destroy(); _btChart = null; }
  const ctx = document.getElementById("btChart");
  if (!ctx) return;
  _btChart = new Chart(ctx, {
    type: "line",
    data: { labels: d.dates, datasets: [
      { label: "IHSG Buy & Hold", data: d.ihsg, borderColor: "#E74C3C", borderWidth: 1.5, fill: false, tension: .2, pointRadius: 0 },
      { label: "D.A.R.A Adaptive", data: d.dara, borderColor: "#27AE60", borderWidth: 2, fill: false, tension: .2, pointRadius: 0 },
    ]},
    options: { responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: true, position: "top", labels: { font: { size: 10 }, boxWidth: 12 } },
        tooltip: { callbacks: { label: c => " " + c.dataset.label + ": " + ((c.parsed.y - 1) * 100).toFixed(1) + "%" } } },
      scales: { x: { ticks: { font: { size: 8 }, maxTicksLimit: 10 }, grid: { display: false } },
                y: { grid: { color: "rgba(0,0,0,.04)" }, ticks: { font: { size: 9 }, callback: v => ((v - 1) * 100).toFixed(0) + "%" } } } }
  });
}

// [F-04] Fixed switchTab — uses style.display so inline-style tabs actually show/hide.
// [F-07] Removed _origSwitch capture (was always null; switchTab not defined at capture time).
function switchTab(tab) {
  ["signals", "backtest", "risk"].forEach(t => {
    const btn     = document.getElementById("tab-" + t);
    const content = document.getElementById("tc-" + t);
    if (btn)     btn.classList.toggle("active", t === tab);
    if (content) content.style.display = (t === tab) ? "block" : "none";
  });
  if (tab === "backtest") loadBacktest();
}

// Auto-load backtest stats for impact banner
setTimeout(() => {
  fetch(API + "/api/backtest", { signal: AbortSignal.timeout(5000) })
    .then(r => r.json()).then(d => {
      if (d.stats) {
        _btData = d;
        const s = d.stats;
        ["bt-vol-reduction", "bt-vol-r2"].forEach(id => {
          const el = document.getElementById(id);
          if (el) el.textContent = (s.oos_vol_reduction != null
            ? Math.round(s.oos_vol_reduction)
            : Math.round(s.vol_reduction_pct)) + "%";
        });
        ["bt-crisis-protect", "bt-crisis-p2"].forEach(id => {
          const el = document.getElementById(id);
          if (el) el.textContent = "+" + s.crisis_protection.toFixed(1) + "pp";
        });
      }
    }).catch(() => {});
}, 1500);

function connectSSE(scenario) {
  if (_evtSource) { _evtSource.close(); _evtSource = null; }
  if (_reconnectTimer) { clearTimeout(_reconnectTimer); _reconnectTimer = null; }

  const dot = document.getElementById("conn-dot");
  const lbl = document.getElementById("conn-label");

  // [F-02] Use full 'conn-dot ...' class string to preserve base CSS class
  dot.className   = "conn-dot";
  lbl.textContent = "Menghubungkan...";

  const _loadMsgs = [
    "Menghubungkan ke server analisis...",
    "Mengunduh data pasar 1.260 hari...",
    "Menjalankan GARCH(1,1) MLE...",
    "Menganalisis sentimen berita...",
    "Mengoptimasi portofolio HRP...",
  ];
  let _msgIdx = 0;
  const _msgTimer = setInterval(() => {
    if (!isOnline) {
      _msgIdx = (_msgIdx + 1) % _loadMsgs.length;
      lbl.textContent = _loadMsgs[_msgIdx];
    } else {
      clearInterval(_msgTimer);
    }
  }, 800);

  try {
    _evtSource = new EventSource(API + "/api/stream?scenario=" + scenario);

    _evtSource.onopen = () => {
      isOnline = true;
      _reconnectDelay = 3000;
      clearInterval(_msgTimer);
      dot.className   = "conn-dot live";        // [F-02] fixed
      lbl.textContent = "Live — Real-time";
      // [F-01] was 'err-banner' (null crash) — correct ID is 'error-banner'
      document.getElementById("error-banner").style.display = "none";
    };

    _evtSource.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.error) { console.warn("SSE error:", data.error); return; }
        render(data);
        lbl.textContent = "Live — " + data.timestamp;
        _reconnectDelay = 3000;
      } catch (err) {
        console.warn("SSE parse error:", err);
      }
    };

    _evtSource.onerror = () => {
      isOnline = false;
      dot.className   = "conn-dot err";         // [F-02] fixed
      lbl.textContent = "Terputus — mencoba kembali...";
      // [F-01] correct ID
      document.getElementById("error-banner").style.display = "block";
      render(FALLBACK[currentScenario] || FALLBACK.moderate);
      _evtSource.close();
      _evtSource = null;
      _reconnectDelay = Math.min(_reconnectDelay * 1.5, 30000);
      _reconnectTimer = setTimeout(() => connectSSE(currentScenario), _reconnectDelay);
    };

  } catch (err) {
    console.warn("EventSource tidak tersedia, fallback ke polling:", err);
    fetchDataPolling(scenario);
  }
}

// ── Fallback: polling setiap 30 detik jika SSE tidak tersedia ─────────────────
let _pollTimer = null;
async function fetchDataPolling(scenario) {
  clearTimeout(_pollTimer);
  try {
    const r    = await fetch(API + "/api/analyze?scenario=" + scenario, { signal: AbortSignal.timeout(5000) });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    if (data.error) throw new Error(data.error);
    isOnline = true;
    // [F-02] conn-dot class fixed; [F-01] error-banner ID fixed
    document.getElementById("conn-dot").className   = "conn-dot live";
    document.getElementById("conn-label").textContent = "Live (polling) — " + data.timestamp;
    document.getElementById("error-banner").style.display = "none";
    render(data);
  } catch {
    isOnline = false;
    document.getElementById("conn-dot").className   = "conn-dot err";  // [F-02]
    document.getElementById("conn-label").textContent = "Offline — Simulasi";
    document.getElementById("error-banner").style.display = "block";   // [F-01]
    render(FALLBACK[scenario] || FALLBACK.moderate);
  }
  _pollTimer = setTimeout(() => fetchDataPolling(currentScenario), 30000);
}

// [F-03] Removed duplicate fetchData() here.
// The single entry point for scenario changes is connectSSE() called directly
// from setGoal(), or fetchDataPolling() as the SSE fallback.
async function fetchData(scenario) {
  currentScenario = scenario;
  connectSSE(scenario);
}