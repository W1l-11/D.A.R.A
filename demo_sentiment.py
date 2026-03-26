import os, json, re, time, logging
from datetime import datetime
from typing import Optional

import requests

log = logging.getLogger(__name__)

POSITIVE_KEYWORDS = {
    # Makro positif
    "growth": 0.8, "tumbuh": 0.8, "naik": 0.7, "rally": 0.9,
    "record": 0.8, "rekor": 0.8, "surplus": 0.7, "recovery": 0.8,
    "optimism": 0.8, "strong": 0.6, "bullish": 0.9, "surge": 0.8,
    # BI & pasar modal
    "cut rate": 0.9, "pivot": 0.9, "dovish": 0.8, "inflow": 0.8,
    "fdi": 0.7, "investment": 0.6, "ipo": 0.6, "tembus": 0.7,
    # Fundamental
    "profit": 0.7, "laba": 0.7, "earnings beat": 0.8, "gdp beat": 0.8,
}

NEGATIVE_KEYWORDS = {
    # Makro negatif
    "crisis": 0.9, "krisis": 0.9, "crash": 0.9, "collapse": 0.9,
    "recession": 0.8, "fall": 0.6, "turun": 0.6, "drop": 0.7,
    "deficit": 0.6, "outflow": 0.8, "selloff": 0.9, "bearish": 0.9,
    # BI & pasar modal
    "rate hike": 0.8, "hawkish": 0.8, "tightening": 0.7,
    "capital flight": 0.9, "net sell": 0.7, "jual": 0.5,
    # Risiko
    "geopolitical": 0.6, "war": 0.8, "perang": 0.8, "sanctions": 0.7,
    "inflation surge": 0.8, "stagflation": 0.9, "default": 0.9,
    # Rupiah
    "rupiah weakens": 0.8, "rupiah falls": 0.8, "devaluation": 0.9,
}


def score_headline(text: str) -> float:
    text_lower = text.lower()

    pos_score = sum(w for kw, w in POSITIVE_KEYWORDS.items() if kw in text_lower)
    neg_score = sum(w for kw, w in NEGATIVE_KEYWORDS.items() if kw in text_lower)

    total = pos_score + neg_score
    if total == 0:
        return 0.0

    raw = (pos_score - neg_score) / total
    return float(max(-1.0, min(1.0, raw * 1.5)))


def fetch_rss_headlines(max_items: int = 15) -> list[dict]:
    feeds = [
        # Reuters Business
        "https://feeds.reuters.com/reuters/businessNews",
        # Yahoo Finance Indonesia
        "https://finance.yahoo.com/news/rss",
        # Investing.com Indonesia
        "https://id.investing.com/rss/news.rss",
    ]

    headlines = []
    seen = set()

    for url in feeds:
        try:
            r = requests.get(url, timeout=8, headers={
                "User-Agent": "Mozilla/5.0 D.A.R.A/3.0"
            })
            if r.status_code != 200:
                continue

            items = re.findall(r'<item>(.*?)</item>', r.text, re.DOTALL)
            for item in items[:max_items]:
                title_match = re.search(r'<title><!\[CDATA\[(.*?)\]\]></title>', item)
                if not title_match:
                    title_match = re.search(r'<title>(.*?)</title>', item)
                if not title_match:
                    continue

                title = title_match.group(1).strip()
                title = re.sub(r'<[^>]+>', '', title)  # strip HTML tags
                if title and title not in seen and len(title) > 10:
                    seen.add(title)
                    headlines.append({
                        "title":  title,
                        "source": url.split('/')[2],
                        "score":  score_headline(title),
                    })

        except Exception as exc:
            log.debug("RSS fetch error [%s]: %s", url, exc)
            continue

    return headlines[:max_items]


def run_demo_pipeline() -> dict:
    headlines = fetch_rss_headlines(max_items=15)

    if not headlines:
        log.info("RSS tidak tersedia — pakai keyword-based demo headlines")
        demo_hl = [
            "Bank Indonesia holds rates steady amid global uncertainty",
            "IHSG extends gains as foreign inflows return to Indonesian markets",
            "Rupiah strengthens against dollar on positive trade data",
            "Indonesia GDP growth meets expectations at 5.1% in Q3",
            "Coal prices support Indonesian commodity exports outlook",
        ]
        headlines = [{"title": h, "source": "demo", "score": score_headline(h)}
                     for h in demo_hl]

    scores = [h["score"] for h in headlines]
    avg    = float(sum(scores) / len(scores))
    ppos   = sum(1 for s in scores if s > 0.1) / len(scores)
    pneg   = sum(1 for s in scores if s < -0.1) / len(scores)
    pneu   = 1 - ppos - pneg
    fg     = (avg + 1) / 2   # fear_greed_index: 0 = extreme fear, 1 = extreme greed

    if avg > 0.2:
        phase = "BULL"
    elif avg < -0.2:
        phase = "BEAR"
    else:
        phase = "NEUTRAL"

    top3 = sorted(headlines, key=lambda x: abs(x["score"]), reverse=True)[:3]

    return {
        "timestamp":              datetime.utcnow().isoformat(),
        "source":                 "rss_rule_based",
        "n_articles_total":       len(headlines),
        "n_articles_relevant":    len(headlines),
        "avg_sentiment_score":    round(avg, 6),
        "sentiment_volatility":   round(float(
            (sum((s - avg)**2 for s in scores) / len(scores))**0.5), 6),
        "pct_positive_news":      round(ppos, 4),
        "pct_negative_news":      round(pneg, 4),
        "pct_neutral_news":       round(pneu, 4),
        "fear_greed_index":       round(fg, 4),
        "pct_bull_phase":         round(ppos, 4),
        "pct_bear_phase":         round(pneg, 4),
        "dominant_market_phase":  phase,
        "top_headlines":          [h["title"] for h in top3],
        "gmm_bic":                None,
    }


def get_live_headlines_for_dara(scenario: Optional[str] = None) -> dict:
    signal = run_demo_pipeline()

    score  = signal["avg_sentiment_score"]
    phase  = signal["dominant_market_phase"]
    fg     = signal["fear_greed_index"]
    total  = signal["n_articles_total"]
    pneg   = signal["pct_negative_news"]
    ppos   = signal["pct_positive_news"]

    real_hl = signal.get("top_headlines", [])
    if len(real_hl) >= 3:
        headlines = real_hl
    else:
        if phase == "BEAR":
            headlines = [
                f"Sentimen BEAR dari {total} artikel berita (RSS live)",
                f"Fear/Greed Index: {fg:.2f} — tekanan jual mendominasi",
                f"Berita negatif: {pneg:.0%} dari total artikel",
            ]
        elif phase == "BULL":
            headlines = [
                f"Sentimen BULL dari {total} artikel berita (RSS live)",
                f"Fear/Greed Index: {fg:.2f} — greed mendominasi",
                f"Berita positif: {ppos:.0%} dari total artikel",
            ]
        else:
            headlines = [
                f"Sentimen NETRAL dari {total} artikel berita (RSS live)",
                f"Fear/Greed Index: {fg:.2f}",
                f"Positif {ppos:.0%} | Negatif {pneg:.0%}",
            ]

    return {
        "market":    float(max(min(score, 1.0), -1.0)),
        "headlines": headlines[:3],
        "_signal":   signal,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  D.A.R.A — Demo Sentiment (RSS + Rule-based)")
    print("=" * 60)

    result = run_demo_pipeline()

    print(f"\n  Source        : {result['source']}")
    print(f"  Articles      : {result['n_articles_total']}")
    print(f"  Avg Sentiment : {result['avg_sentiment_score']:+.4f}")
    print(f"  Fear/Greed    : {result['fear_greed_index']:.2f}")
    print(f"  Phase         : {result['dominant_market_phase']}")
    print(f"  Positive      : {result['pct_positive_news']:.1%}")
    print(f"  Negative      : {result['pct_negative_news']:.1%}")
    print(f"\n  Top Headlines:")
    for h in result.get("top_headlines", []):
        print(f"    • {h[:70]}")
    print()
