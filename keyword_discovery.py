import os, json, time, logging, warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

NEWS_API_KEY    = os.getenv("NEWS_API_KEY", "")
DAYS_BACK       = 28
TOP_N_KEYWORDS  = 50
MIN_CORRELATION = 0.20
OUTPUT_DIR      = Path(".")

TICKERS = {"IHSG": "^JKSE", "USD/IDR": "USDIDR=X"}

SEED_TOPICS = [
    "Bank Indonesia monetary policy interest rate",
    "Indonesia rupiah exchange rate currency",
    "Indonesia inflation consumer price index",
    "Indonesia GDP economic growth outlook",
    "Indonesia government budget deficit spending",
    "Indonesia trade balance current account",
    "IHSG Jakarta stock exchange IDX",
    "Indonesia stock market equities investment",
    "Indonesia government bonds SBN ORI sukuk",
    "Indonesia foreign investor capital flow",
    "coal price export Indonesia energy",
    "palm oil CPO price Indonesia",
    "nickel price Indonesia mining battery",
    "Indonesia banking sector BRI BCA Mandiri",
    "Indonesia foreign direct investment FDI",
    "Federal Reserve rate decision dollar emerging",
    "China economy slowdown trade Asia",
    "emerging markets capital outflow selloff",
    "geopolitical risk Asia Pacific tension",
    "Indonesia sovereign credit rating Moody Fitch",
]

EXTRA_STOPWORDS = [
    "said","year","new","news","say","says","market","percent","will","may",
    "could","would","also","one","two","first","million","billion","president",
    "government","week","monday","tuesday","wednesday","thursday","friday",
    "january","february","march","april","2024","2025","2026",
]


def _default_keywords() -> list:
    return [
        "Indonesia interest rate","Bank Indonesia","rupiah","IHSG",
        "inflation Indonesia","Fed rate","coal Indonesia","palm oil",
        "nickel price","capital outflow","foreign investor","SBN",
        "GDP Indonesia","recession","oil prices","dollar strengthens",
        "emerging markets","credit rating Indonesia",
    ]


def _save_keywords(keywords: list, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "query_keywords.json", "w") as f:
        json.dump(keywords, f, indent=2, ensure_ascii=False)
    log.info("Keywords disimpan: %d entries", len(keywords))


def load_keywords(path: str = "query_keywords.json") -> list:
    try:
        with open(path) as f:
            kw = json.load(f)
        if kw:
            log.info("Loaded %d keywords dari %s", len(kw), path)
            return kw
    except Exception:
        pass
    log.info("query_keywords.json belum ada — pakai keyword default (%d)", len(_default_keywords()))
    return _default_keywords()


def fetch_articles(api_key, query, start, end, page_size=100):
    if not api_key:
        return []
    try:
        r = requests.get("https://newsapi.org/v2/everything", params={
            "q": query, "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sortBy": "publishedAt", "language": "en", "pageSize": page_size,
            "domains": "reuters.com,bloomberg.com,ft.com,cnbc.com,thejakartapost.com,investing.com",
            "apiKey": api_key,
        }, timeout=15)
        r.raise_for_status()
        return [{"title": (a.get("title") or "").strip(),
                 "description": (a.get("description") or "").strip(),
                 "published_at": a.get("publishedAt",""),
                 "seed_topic": query}
                for a in r.json().get("articles",[]) if (a.get("title") or "").strip()]
    except Exception as exc:
        log.warning("NewsAPI [%s]: %s", query[:30], exc)
        return []


def score_finbert(df):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        log.info("Loading FinBERT...")
        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        mdl.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(device)

        @torch.no_grad()
        def _s(title, desc=""):
            text   = (title + ". " + desc)[:900]
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
            probs  = torch.softmax(mdl(**inputs).logits, dim=-1).cpu().numpy()[0]
            lmap   = mdl.config.id2label
            lp     = {lmap[i]: float(probs[i]) for i in range(len(probs))}
            return lp.get("positive",0) - lp.get("negative",0)

        scores = []
        for i, row in df.iterrows():
            scores.append(_s(row["title"], row.get("description","")))
            if len(scores) % 20 == 0: log.info("  FinBERT: %d/%d", len(scores), len(df))
        df["sentiment_score"] = scores
    except Exception as exc:
        log.warning("FinBERT fallback: %s", exc)
        POS = ["growth","record","high","surge","rally","profit","strong","optimism","recovery"]
        NEG = ["fall","drop","crisis","crash","loss","weak","risk","fear","recession","inflation"]
        def _h(text):
            t = text.lower()
            return (sum(w in t for w in POS) - sum(w in t for w in NEG)) * 0.15
        df["sentiment_score"] = df.apply(lambda r: _h(r["title"]+" "+str(r.get("description",""))), axis=1)
    return df


def run_discovery(api_key=None, days_back=None, min_corr=None, output_dir=None) -> list:
    from sklearn.feature_extraction.text import TfidfVectorizer

    key  = api_key   or NEWS_API_KEY
    days = days_back or DAYS_BACK
    mc   = min_corr  or MIN_CORRELATION
    od   = output_dir or OUTPUT_DIR

    log.info("=" * 55)
    log.info("  D.A.R.A Keyword Discovery  |  %d hari terakhir", days)
    log.info("=" * 55)

    if not key:
        log.warning("NEWS_API_KEY tidak ada — pakai keyword default")
        kw = _default_keywords(); _save_keywords(kw, od); return kw

    end, start = datetime.utcnow(), datetime.utcnow() - timedelta(days=days)
    all_arts, seen = [], set()
    for topic in SEED_TOPICS:
        for a in fetch_articles(key, topic, start, end, 100):
            if a["title"] not in seen:
                seen.add(a["title"]); all_arts.append(a)
        time.sleep(0.4)

    if not all_arts:
        log.warning("Tidak ada artikel — pakai default"); kw = _default_keywords(); _save_keywords(kw, od); return kw

    news_df = pd.DataFrame(all_arts)
    news_df["published_at"] = pd.to_datetime(news_df["published_at"], utc=True, errors="coerce")
    news_df["date"]         = news_df["published_at"].dt.date
    news_df = news_df.dropna(subset=["published_at","title"])
    news_df["full_text"]    = news_df["title"] + ". " + news_df.get("description", pd.Series("", index=news_df.index)).fillna("")
    log.info("Total artikel: %d", len(news_df))

    # TF-IDF
    vec = TfidfVectorizer(max_features=600, ngram_range=(1,2), stop_words="english",
                          min_df=2, max_df=0.80, sublinear_tf=True)
    mat   = vec.fit_transform(news_df["full_text"])
    names = vec.get_feature_names_out()
    tfidf = pd.Series(np.asarray(mat.mean(axis=0)).flatten(), index=names)
    tfidf = tfidf[~tfidf.index.str.lower().isin(EXTRA_STOPWORDS)].sort_values(ascending=False)
    top_kw = list(tfidf.head(TOP_N_KEYWORDS).index)
    log.info("Top TF-IDF keywords: %s", top_kw[:8])

    # FinBERT
    news_df = score_finbert(news_df)

    # Market data
    market_returns = {}
    try:
        import yfinance as yf
        for name, ticker in TICKERS.items():
            raw = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                              end=(end+timedelta(days=1)).strftime("%Y-%m-%d"),
                              auto_adjust=True, progress=False)
            ret = raw["Close"].squeeze().pct_change().dropna()
            ret.index = pd.to_datetime(ret.index).date
            market_returns[name] = ret
    except Exception as exc:
        log.warning("Market data gagal: %s", exc)

    if not market_returns:
        kw = top_kw[:20]; _save_keywords(kw, od); return kw

    tagged = []
    for _, row in news_df.iterrows():
        tl = row["full_text"].lower()
        for kw in top_kw:
            if kw.lower() in tl:
                tagged.append({"keyword":kw,"date":row["date"],"sentiment_score":row.get("sentiment_score",0)})
    if not tagged:
        kw = top_kw[:20]; _save_keywords(kw, od); return kw

    sent_daily = (pd.DataFrame(tagged)
                  .groupby(["date","keyword"])["sentiment_score"].mean()
                  .reset_index().pivot(index="date",columns="keyword",values="sentiment_score"))
    sent_daily.index = pd.to_datetime(sent_daily.index).date

    corr_rows = []
    for mkt_name, returns in market_returns.items():
        returns.index = pd.to_datetime(returns.index).date
        for kw in sent_daily.columns:
            s      = sent_daily[kw].dropna()
            common = sorted(set(s.index) & set(returns.index))
            if len(common) < 3: continue
            r_val, p_val = stats.pearsonr(s.loc[common].values, returns.loc[common].values)
            corr_rows.append({"keyword":kw,"market":mkt_name,
                               "correlation":round(r_val,4),"p_value":round(p_val,4),
                               "significant":p_val<0.05,"n_days":len(common),
                               "tfidf_score":round(float(tfidf.get(kw,0)),4)})

    if not corr_rows:
        kw = top_kw[:20]; _save_keywords(kw, od); return kw

    corr_df = pd.DataFrame(corr_rows).sort_values("correlation", key=abs, ascending=False)
    best    = (corr_df.groupby("keyword")
               .apply(lambda x: x.loc[x["correlation"].abs().idxmax()])
               .reset_index(drop=True)
               .sort_values("correlation", key=abs, ascending=False))
    strong  = best[best["correlation"].abs() >= mc]

    final_kw = strong["keyword"].tolist()
    if len(final_kw) < 10:
        extras = best[best["correlation"].abs() < mc].head(10-len(final_kw))["keyword"].tolist()
        final_kw.extend(extras)

    _save_keywords(final_kw, od)
    corr_df.to_csv(od / "discovered_keyword_correlations.csv", index=False)
    best.to_csv(od / "recommended_keywords.csv", index=False)

    log.info("=" * 55)
    log.info("  SELESAI: %d keyword kuat ditemukan", len(final_kw))
    for kw in final_kw[:10]:
        row = best[best["keyword"]==kw]
        if not row.empty:
            log.info("  %-32s r=%+.3f", kw, row.iloc[0]["correlation"])
    log.info("=" * 55)
    return final_kw


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  D.A.R.A — Keyword Discovery")
    print("="*55)
    if not NEWS_API_KEY:
        print("\n⚠  NEWS_API_KEY tidak ditemukan di .env")
        print("   Menyimpan keyword default...\n")
        kw = _default_keywords()
        _save_keywords(kw)
    else:
        kw = run_discovery()
    print(f"\n✅ {len(kw)} keyword disimpan ke query_keywords.json:")
    for i, k in enumerate(kw, 1):
        print(f"  {i:2}. {k}")
