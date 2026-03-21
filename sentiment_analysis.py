"""
=============================================================================
  FINANCIAL NEWS SENTIMENT ANALYSIS
  FinBERT + Gaussian Mixture Model (Bull/Bear Phase Detection)
  Output: Sentiment features for GARCH volatility prediction
=============================================================================
"""

import os
import json
import warnings
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

load_dotenv(verbose=True)  # reads .env in project root
# ===========================================================================
# CONFIG  (edit these to match your project)
# ===========================================================================
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY_HERE")

# Topics / tickers from keyword discovery model
QUERY_KEYWORDS: list[str] = [
    "Dollar",
    "Inflation",
    "Strait Hormuz",
    "Global conflict",
    "far",  
    "hit", 
    "asia",  
    "markets", 
    "energy", 
    "oil prices", 
    "strait",  
    "hormuz",  
    "prices",  
    "conflict", 
    "growth",  
    "iran war", 
    "east",  
    "rates",  
    "price",  
    "middle east",  
    "middle",  
    "iran",  
    "oil",  
    "start",  
    "pressures",  
    "war",  
    "fuel",  
    "shock",  
    "spike",  
    "india",  
    "amid"
]

# Number of articles to pull per keyword (max 5 on free tier → 5 × 5 = 25 headlines)
ARTICLES_PER_QUERY: int = 6

# GMM: number of market phases to detect (2 = bull / bear; 3 adds neutral)
N_GMM_COMPONENTS: int = 2

# Threshold for classifying a single article as market-relevant (0–1)
RELEVANCE_SCORE_THRESHOLD: float = 0.55

# Output CSV path (empty string → skip saving)
OUTPUT_CSV: str = "sentiment_output.csv"

# ===========================================================================
# FALLBACK HEADLINES  (used when NewsAPI key is not set / quota exceeded)
# ===========================================================================

DEMO_HEADLINES: list[str] = [
    "Federal Reserve signals aggressive rate hikes to combat persistent inflation",
    "S&P 500 plunges 3% as recession fears grip Wall Street",
    "Tech stocks surge after better-than-expected earnings season results",
    "Unemployment falls to 3.4%, boosting consumer confidence and market rally",
    "Oil prices spike amid Middle East tensions, stoking inflation concerns",
    "China GDP growth slows sharply, rattling global equity markets",
    "Bitcoin tumbles 18% following stricter crypto regulation announcements",
    "Strong retail sales data suggest economy remains resilient despite rate hikes",
    "Dow Jones hits record high as soft landing scenario gains credibility",
    "Banking sector turmoil deepens as regional lender shares collapse",
    "ECB raises rates to highest level in two decades amid Eurozone inflation",
    "US Treasury yields invert further, signalling deeper recession risk",
    "Apple posts record quarterly revenue, lifting broader market sentiment",
    "IMF warns of synchronised global slowdown in latest economic outlook",
    "Fed Chair signals potential rate cut as inflation shows signs of cooling",
    "Geopolitical tensions escalate, sending gold to 6-month high",
    "Job market cools: payrolls miss forecast, market cheers potential Fed pause",
    "Corporate defaults rising rapidly as high interest rates squeeze margins",
    "Nvidia revenue triples on AI chip demand, sparking tech sector rally",
    "Consumer spending unexpectedly drops, raising stagflation fears on Wall Street",
]


# ===========================================================================
# 1.  NEWS FETCHER
# ===========================================================================
class NewsAPIFetcher:
    """Fetch financial headlines from NewsAPI."""

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._valid = bool(api_key) 

    # ------------------------------------------------------------------
    def fetch(
        self,
        queries: list[str],
        articles_per_query: int = 5,
        days_back: int = 3,
    ) -> list[dict]:
        """Return a deduplicated list of article dicts with title + description."""
        if not self._valid:
            log.warning("NewsAPI key missing → using demo headlines.")
            return self._demo_articles()

        from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        seen: set[str] = set()
        articles: list[dict] = []

        for q in queries:
            params = {
                "q": q,
                "from": from_date,
                "sortBy": "relevancy",
                "language": "en",
                "pageSize": articles_per_query,
                "apiKey": self.api_key,
            }
            try:
                r = requests.get(self.BASE_URL, params=params, timeout=10)
                r.raise_for_status()
                for art in r.json().get("articles", []):
                    title = (art.get("title") or "").strip()
                    if title and title not in seen:
                        seen.add(title)
                        articles.append(
                            {
                                "title": title,
                                "description": (art.get("description") or "").strip(),
                                "source": art.get("source", {}).get("name", ""),
                                "published_at": art.get("publishedAt", ""),
                                "query": q,
                            }
                        )
            except Exception as exc:
                log.error("NewsAPI error for query '%s': %s", q, exc)

        log.info("Fetched %d unique headlines from NewsAPI.", len(articles))
        return articles if articles else self._demo_articles()

    # ------------------------------------------------------------------
    @staticmethod
    def _demo_articles() -> list[dict]:
        return [
            {
                "title": h,
                "description": "",
                "source": "DEMO",
                "published_at": datetime.utcnow().isoformat(),
                "query": "demo",
            }
            for h in DEMO_HEADLINES
        ]


# ===========================================================================
# 2.  FINBERT SENTIMENT ENGINE
# ===========================================================================
class FinBERTEngine:
    """
    Wraps ProsusAI/finbert for financial sentiment classification.
    Labels: positive, negative, neutral → mapped to +1, -1, 0.
    """

    MODEL_NAME = "ProsusAI/finbert"
    LABEL_MAP = {"positive": 1, "negative": -1, "neutral": 0}

    def __init__(self):
        log.info("Loading FinBERT model (first run downloads ~440 MB) …")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        log.info("FinBERT loaded on %s.", self.device)

    # ------------------------------------------------------------------
    def _build_text(self, title: str, description: str) -> str:
        """Combine title + description, truncated to 512 tokens worth of chars."""
        combined = title
        if description:
            combined += ". " + description
        return combined[:900]  # rough char limit before tokenisation

    # ------------------------------------------------------------------
    @torch.no_grad()
    def analyse(self, articles: list[dict]) -> list[dict]:
        """Return articles enriched with sentiment scores."""
        results = []
        for art in articles:
            text = self._build_text(art["title"], art.get("description", ""))

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            # FinBERT label order: positive=0, negative=1, neutral=2
            # (verify with model.config.id2label if model changes)
            id2label = self.model.config.id2label  # {0: 'positive', 1: 'negative', 2: 'neutral'}
            label_probs = {id2label[i]: float(probs[i]) for i in range(len(probs))}

            dominant_label = max(label_probs, key=label_probs.get)
            dominant_score = label_probs[dominant_label]

            # Continuous sentiment score in [-1, +1]
            sentiment_score = (
                label_probs.get("positive", 0.0)
                - label_probs.get("negative", 0.0)
            )

            results.append(
                {
                    **art,
                    "prob_positive": label_probs.get("positive", 0.0),
                    "prob_negative": label_probs.get("negative", 0.0),
                    "prob_neutral": label_probs.get("neutral", 0.0),
                    "sentiment_label": dominant_label,
                    "sentiment_confidence": dominant_score,
                    "sentiment_score": round(sentiment_score, 6),  # KEY feature for GMM
                    "is_relevant": dominant_score >= RELEVANCE_SCORE_THRESHOLD,
                }
            )

        log.info("FinBERT analysis complete for %d articles.", len(results))
        return results


# ===========================================================================
# 3.  GAUSSIAN MIXTURE MODEL — Bull / Bear Phase Detector
# ===========================================================================
class MarketPhaseGMM:
    """
    Fits a GMM on sentiment scores to identify bull / bear (/ neutral) regimes.

    Features fed to GMM:
      - sentiment_score        continuous in [-1, +1]
      - prob_positive          raw positive probability
      - prob_negative          raw negative probability
      - sentiment_confidence   how certain FinBERT is about the dominant label
    """

    def __init__(self, n_components: int = N_GMM_COMPONENTS):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            n_init=10,
            random_state=42,
        )
        self.phase_labels: dict[int, str] = {}

    # ------------------------------------------------------------------
    def _feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        cols = ["sentiment_score", "prob_positive", "prob_negative", "sentiment_confidence"]
        return df[cols].values

    # ------------------------------------------------------------------
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._feature_matrix(df)
        X_scaled = self.scaler.fit_transform(X)
        self.gmm.fit(X_scaled)

        df = df.copy()
        df["gmm_cluster"] = self.gmm.predict(X_scaled)
        df["gmm_probability"] = self.gmm.predict_proba(X_scaled).max(axis=1)

        # ---- Label clusters: cluster with higher mean sentiment_score → bull
        cluster_means = (
            df.groupby("gmm_cluster")["sentiment_score"].mean().sort_values(ascending=False)
        )

        if self.n_components == 2:
            self.phase_labels = {
                cluster_means.index[0]: "BULL",
                cluster_means.index[1]: "BEAR",
            }
        else:
            phase_names = ["BULL", "NEUTRAL", "BEAR"]
            self.phase_labels = {
                cluster_means.index[i]: phase_names[i]
                for i in range(self.n_components)
            }

        df["market_phase"] = df["gmm_cluster"].map(self.phase_labels)
        log.info("GMM cluster → phase mapping: %s", self.phase_labels)
        return df

    # ------------------------------------------------------------------
    def get_aggregate_signal(self, df: pd.DataFrame) -> dict:
        """
        Aggregate all articles into a single signal vector for downstream GARCH input.
        """
        relevant = df[df["is_relevant"]] if "is_relevant" in df.columns else df

        avg_sentiment = relevant["sentiment_score"].mean()
        vol_sentiment = relevant["sentiment_score"].std()
        pct_positive = (relevant["sentiment_label"] == "positive").mean()
        pct_negative = (relevant["sentiment_label"] == "negative").mean()
        pct_bull = (relevant["market_phase"] == "BULL").mean() if "market_phase" in relevant.columns else np.nan
        pct_bear = (relevant["market_phase"] == "BEAR").mean() if "market_phase" in relevant.columns else np.nan

        # Dominant phase by weighted GMM probability
        if "gmm_probability" in relevant.columns and len(relevant) > 0:
            dominant_idx = relevant["gmm_probability"].idxmax()
            dominant_phase = relevant.loc[dominant_idx, "market_phase"]
        else:
            dominant_phase = "UNKNOWN"

        # Composite fear/greed index: scaled to [0, 1] where 1 = extreme greed
        fear_greed_index = (avg_sentiment + 1) / 2  # maps [-1,1] → [0,1]

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "n_articles_total": len(df),
            "n_articles_relevant": len(relevant),
            # --- Core GARCH exogenous features ---
            "avg_sentiment_score": round(avg_sentiment, 6),
            "sentiment_volatility": round(vol_sentiment, 6),
            "pct_positive_news": round(pct_positive, 4),
            "pct_negative_news": round(pct_negative, 4),
            "fear_greed_index": round(fear_greed_index, 4),
            # --- Phase info ---
            "pct_bull_phase": round(pct_bull, 4),
            "pct_bear_phase": round(pct_bear, 4),
            "dominant_market_phase": dominant_phase,
            # --- GMM model meta ---
            "gmm_bic": round(self.gmm.bic(
                self.scaler.transform(self._feature_matrix(df))
            ), 2),
        }


# ===========================================================================
# 4.  PRETTY REPORT
# ===========================================================================
def print_report(df: pd.DataFrame, signal: dict) -> None:
    WIDTH = 72
    SEP = "─" * WIDTH

    print(f"\n{'═' * WIDTH}")
    print(f"  FINANCIAL SENTIMENT ANALYSIS REPORT")
    print(f"  Generated: {signal['timestamp']}")
    print(f"{'═' * WIDTH}\n")

    # ---- Per-article table -----------------------------------------------
    print(f"{'#':<3}  {'SENTIMENT':<10}  {'SCORE':>7}  {'PHASE':<8}  HEADLINE")
    print(SEP)

    PHASE_ICON = {"BULL": "📈", "BEAR": "📉", "NEUTRAL": "➖", "UNKNOWN": "❓"}
    SENT_ICON  = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}

    for i, row in df.iterrows():
        icon_s = SENT_ICON.get(row["sentiment_label"], "❓")
        icon_p = PHASE_ICON.get(row.get("market_phase", "UNKNOWN"), "❓")
        headline = row["title"][:55] + ("…" if len(row["title"]) > 55 else "")
        conf = f"({row['sentiment_confidence']:.0%})"
        print(
            f"{i+1:<3}  {icon_s} {row['sentiment_label']:<8}{conf:>6}  "
            f"{row['sentiment_score']:>+7.3f}  {icon_p} {row.get('market_phase','?'):<6}  {headline}"
        )

    print(SEP)

    # ---- Aggregate signal -----------------------------------------------
    print(f"\n{'AGGREGATE SIGNAL (GARCH Exogenous Features)':^{WIDTH}}")
    print(SEP)

    phase_icon = PHASE_ICON.get(signal["dominant_market_phase"], "❓")
    bar_len = int(signal["fear_greed_index"] * 40)
    bar = "█" * bar_len + "░" * (40 - bar_len)

    print(f"  Articles analysed  : {signal['n_articles_total']}  "
          f"(relevant: {signal['n_articles_relevant']})")
    print(f"  Avg sentiment score: {signal['avg_sentiment_score']:+.4f}  "
          f"(volatility: {signal['sentiment_volatility']:.4f})")
    print(f"  Positive news      : {signal['pct_positive_news']:.1%}")
    print(f"  Negative news      : {signal['pct_negative_news']:.1%}")
    print(f"  Fear/Greed Index   : [{bar}] {signal['fear_greed_index']:.2f}")
    print(f"  Bull phase %       : {signal['pct_bull_phase']:.1%}")
    print(f"  Bear phase %       : {signal['pct_bear_phase']:.1%}")
    print(f"  Dominant phase     : {phase_icon}  {signal['dominant_market_phase']}")
    print(f"  GMM BIC score      : {signal['gmm_bic']}")
    print(SEP)

    print("\n  📤  GARCH-READY FEATURE VECTOR:")
    garch_keys = [
        "avg_sentiment_score",
        "sentiment_volatility",
        "pct_positive_news",
        "pct_negative_news",
        "fear_greed_index",
        "pct_bull_phase",
        "pct_bear_phase",
    ]
    for k in garch_keys:
        print(f"     {k:<28}: {signal[k]}")

    print(f"\n{'═' * WIDTH}\n")


# ===========================================================================
# 5.  MAIN PIPELINE
# ===========================================================================
def run_pipeline() -> tuple[pd.DataFrame, dict]:
    """
    End-to-end pipeline:
        NewsAPI → FinBERT → GMM → Aggregate signal
    Returns the enriched DataFrame and the aggregate signal dict.
    """
    log.info("=== Starting Sentiment Analysis Pipeline ===")
    
    # --- Step 1: Fetch news -----------------------------------------------
    fetcher = NewsAPIFetcher(api_key=NEWS_API_KEY)
    articles = fetcher.fetch(
        queries=QUERY_KEYWORDS,
        articles_per_query=ARTICLES_PER_QUERY,
    )

    # Limit to 20 articles for demo clarity
    articles = articles[:20]

    # --- Step 2: FinBERT inference ----------------------------------------
    engine = FinBERTEngine()
    enriched = engine.analyse(articles)

    df = pd.DataFrame(enriched)

    # --- Step 3: GMM phase detection --------------------------------------
    gmm = MarketPhaseGMM(n_components=N_GMM_COMPONENTS)
    df = gmm.fit_predict(df)

    # --- Step 4: Aggregate signal for GARCH -------------------------------
    signal = gmm.get_aggregate_signal(df)

    # --- Step 5: Report ---------------------------------------------------
    print_report(df, signal)

    # --- Step 6: Save output ----------------------------------------------
    if OUTPUT_CSV:
        out_cols = [
            "title", "source", "published_at",
            "sentiment_label", "sentiment_score", "sentiment_confidence",
            "prob_positive", "prob_negative", "prob_neutral",
            "is_relevant", "market_phase", "gmm_cluster", "gmm_probability",
        ]
        df[out_cols].to_csv(OUTPUT_CSV, index=False)
        log.info("Article-level results saved → %s", OUTPUT_CSV)

        signal_file = OUTPUT_CSV.replace(".csv", "_signal.json")
        with open(signal_file, "w") as f:
            json.dump(signal, f, indent=2)
        log.info("Aggregate signal saved → %s", signal_file)

    return df, signal


# ===========================================================================
# 6.  REAL-TIME LOOP  (optional — uncomment to run continuously)
# ===========================================================================
def run_realtime(interval_minutes: int = 15) -> None:
    """
    Poll NewsAPI every `interval_minutes` and refresh the sentiment signal.
    The signal dict is ready to be consumed by your GARCH model.
    """
    import time

    log.info("Real-time mode: refreshing every %d minutes.", interval_minutes)
    while True:
        try:
            _, signal = run_pipeline()
            # ---------------------------------------------------------------
            # TODO: pass `signal` to your GARCH model here, e.g.:
            #   garch_model.update_exogenous(signal)
            # ---------------------------------------------------------------
        except Exception as exc:
            log.error("Pipeline error: %s", exc, exc_info=True)

        log.info("Next refresh in %d minutes …", interval_minutes)
        time.sleep(interval_minutes * 60)


# ===========================================================================
if __name__ == "__main__":
    df_result, agg_signal = run_pipeline()

    # Uncomment below to run in real-time polling mode instead:
    # run_realtime(interval_minutes=15)
