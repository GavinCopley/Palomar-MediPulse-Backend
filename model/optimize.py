"""
ML-powered video optimiser  (v2 – semantic text features, cleaned)
─────────────────────────────────────────────────────────────────
Target (y)  = 0.6*view_rate + 0.3*like_rate + 0.1*comment_rate
Numeric     = length_sec, title_len, desc_len, tag_count, is_hd,
              has_captions, category_id, publish_dow, publish_hour
Text        = Sentence-Transformer embedding of title + description
Model       = [Numeric ‖ SVD(embeddings)] → GradientBoostingRegressor
"""

from __future__ import annotations
import os, json, glob, warnings, time
from pathlib import Path

# ─── ENV / CONFIG ─────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv("Gemini_optimize.env", override=False)

DATA_DIR    = Path("data/yt")
MODEL_FPATH = Path("video_model.pkl")
CACHE_FPATH = Path("data/gemini_cache.json")
CACHE_FPATH.parent.mkdir(parents=True, exist_ok=True)

IMPROVEMENT_FACTOR = float(os.getenv("IMPROVEMENT_FACTOR", "0.1"))

# feature switch
MODULE_USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "1") != "0"
EMBED_MODEL_NAME      = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
TEXT_SVD_DIM          = int(os.getenv("TEXT_SVD_DIM", "100"))

# ─── libraries ────────────────────────────────────────────────────────────
import joblib, pandas as pd, numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute   import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

if MODULE_USE_EMBEDDINGS:
    try:
        from sentence_transformers import SentenceTransformer    # noqa: E402
    except ImportError:
        MODULE_USE_EMBEDDINGS = False

if not MODULE_USE_EMBEDDINGS:
    from sklearn.feature_extraction.text import TfidfVectorizer   # noqa: E402

# ─── Gemini (optional) ────────────────────────────────────────────────────
USE_GEMINI   = os.getenv("USE_GEMINI", "1") != "0"
GEMINI_MODEL = None
if USE_GEMINI and (key := os.getenv("GEMINI_API_KEY", "").strip()):
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        GEMINI_MODEL = genai.GenerativeModel(
            os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
        )
    except Exception:
        USE_GEMINI = False

# ─── CSV helpers ──────────────────────────────────────────────────────────
RENAME = {"datepublished":"date published","date_published":"date published",
          "videoid":"video id","video_id":"video id",
          "videotitle":"video title","video_title":"video title",
          "videodescription":"video description","video_description":"video description",
          "durationsec":"duration sec","duration_sec":"duration sec",
          "viewcount":"view count","view_count":"view count",
          "likecount":"like count","like_count":"like count",
          "commentcount":"comment count","comment_count":"comment count"}
BASE_DEFAULTS = {"date published":pd.NaT,"video id":"","video title":"",
                 "video description":"","tags":"","duration sec":0,
                 "view count":0,"like count":0,"comment count":0,
                 "definition":"","captions":"F","category_id":0}

def _load_all_csvs(folder:Path)->pd.DataFrame:
    dfs = []
    for fp in glob.glob(str(folder / "**/*.csv"), recursive=True):
        try:
            df = pd.read_csv(fp)
            df.columns = df.columns.str.lower().str.strip()
            dfs.append(df)
            print("✔︎ read", fp)
        except Exception as e:
            print("✗ skip", fp, "→", e)
    if not dfs:
        raise RuntimeError(f"No CSVs found in {folder}")
    return pd.concat(dfs, ignore_index=True)

def _prep_dataframe(df:pd.DataFrame)->pd.DataFrame:
    for old, new in RENAME.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)
    for col, default in BASE_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    df["length_sec"]   = pd.to_numeric(df["duration sec"], errors="coerce")
    df["title_len"]    = df["video title"].astype(str).str.len()
    df["desc_len"]     = df["video description"].astype(str).str.len()
    df["tag_count"]    = df["tags"].fillna("").astype(str).str.split("|").str.len()
    df["is_hd"]        = df["definition"].str.contains("hd", case=False, na=False).astype(int)
    df["has_captions"] = df["captions"].astype(str).str.startswith(("t","y","1","T","Y")).astype(int)

    dt = pd.to_datetime(df["date published"], errors="coerce")
    df["publish_dow"]  = dt.dt.dayofweek.fillna(0).astype(int)
    df["publish_hour"] = dt.dt.hour.fillna(12).astype(int)

    for c in ["view count", "like count", "comment count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["view count", "like count", "comment count", "length_sec"], inplace=True)

    df["view_rate"]    = df["view count"] / 1000
    df["like_rate"]    = df["like count"] / 1000
    df["comment_rate"] = df["comment count"] / 1000
    return df

NUMERICAL_FEATS = [
    "length_sec", "title_len", "desc_len", "tag_count",
    "is_hd", "has_captions", "category_id", "publish_dow", "publish_hour"
]

# ─── VideoOptimiser ───────────────────────────────────────────────────────
class VideoOptimiser:
    def __init__(self, data_dir:Path=DATA_DIR):
        self.df_raw = _prep_dataframe(_load_all_csvs(data_dir))
        self.df_raw["engagement_score"] = (
            0.6 * self.df_raw["view_rate"] +
            0.3 * self.df_raw["like_rate"] +
            0.1 * self.df_raw["comment_rate"]
        )

        # instance flag (no global tweak)
        self.use_embeddings = MODULE_USE_EMBEDDINGS
        self.embedder = self.tfidf = self.svd = None
        if self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
            except Exception as e:
                print("⚠️  embedding load failed → TF-IDF fallback:", e)
                self.use_embeddings = False

        self._ensure_trained()
        self._cache = self._load_cache()

    # ─── public -----------------------------------------------------------
    def suggest(self, payload:dict, top_n:int=5)->dict:
        X_num  = self._numeric_df(payload).values
        X_text = self._encode([payload.get("title","")+" "+payload.get("description","")])
        X_full = np.hstack([X_num, X_text])
        X_proc = self.scaler.transform(self.imputer.transform(X_full))

        pred      = float(self.model.predict(X_proc)[0])
        improved  = round(pred * (1 + IMPROVEMENT_FACTOR), 2)
        refs      = self._nearest(X_proc, top_n)
        tips      = self._gemini(payload, refs) if USE_GEMINI else {"error": "Gemini disabled"}

        return {"predicted_engagement": round(pred, 2),
                "improved_engagement": improved,
                "reference_videos": refs,
                "gemini_tips": tips}

    # ─── train / load -----------------------------------------------------
    def _ensure_trained(self):
        if MODEL_FPATH.exists():
            obj = joblib.load(MODEL_FPATH)
            self.model   = obj["model"]
            self.imputer = obj["imputer"]
            self.scaler  = obj["scaler"]
            self.svd     = obj.get("svd")
            self.tfidf   = obj.get("tfidf")
        else:
            self._train()

    def _train(self):
        X_num = self.df_raw[NUMERICAL_FEATS].values
        texts = (self.df_raw["video title"].fillna("") + " " +
                 self.df_raw["video description"].fillna("")).tolist()
        X_txt = self._encode(texts)
        X_all = np.hstack([X_num, X_txt])
        y     = self.df_raw["engagement_score"].values

        self.imputer = SimpleImputer(strategy="median").fit(X_all)
        X_imp   = self.imputer.transform(X_all)
        self.scaler = StandardScaler().fit(X_imp)
        X_scaled = self.scaler.transform(X_imp)

        Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
        self.model = GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(Xtr, ytr)

        joblib.dump({"model": self.model, "imputer": self.imputer,
                     "scaler": self.scaler, "svd": self.svd, "tfidf": self.tfidf},
                    MODEL_FPATH)

    # ─── text encoding ----------------------------------------------------
    def _encode(self, texts:list[str])->np.ndarray:
        if self.use_embeddings:
            emb = self.embedder.encode(texts, show_progress_bar=False)
            if self.svd is None:
                self.svd = TruncatedSVD(n_components=TEXT_SVD_DIM,
                                        random_state=42).fit(emb)
            return self.svd.transform(emb)

        # TF-IDF fallback
        if self.tfidf is None:
            self.tfidf = TfidfVectorizer(max_features=500).fit(texts)
            self.svd   = TruncatedSVD(n_components=TEXT_SVD_DIM,
                                      random_state=42).fit(self.tfidf.transform(texts))
        return self.svd.transform(self.tfidf.transform(texts))

    # ─── helpers ----------------------------------------------------------
    def _numeric_df(self, d:dict)->pd.DataFrame:
        return pd.DataFrame([{
            "length_sec"  : d.get("duration_sec"),
            "title_len"   : len(d.get("title","")),
            "desc_len"    : len(d.get("description","")),
            "tag_count"   : len(d.get("tags","").split("|")) if d.get("tags") else 0,
            "is_hd"       : int(bool(d.get("is_hd",1))),
            "has_captions": int(bool(d.get("has_captions",0))),
            "category_id" : d.get("category_id",0),
            "publish_dow" : d.get("publish_dow",0),
            "publish_hour": d.get("publish_hour",12)
        }])

    def _nearest(self, X_scaled, k:int):
        all_scaled = self.scaler.transform(
            self.imputer.transform(
                np.hstack([self.df_raw[NUMERICAL_FEATS].values,
                           self._encode((self.df_raw["video title"]+" "+
                                         self.df_raw["video description"]).tolist())])))
        good = self.df_raw["engagement_score"] > np.percentile(
            self.df_raw["engagement_score"], 80)
        idx  = np.linalg.norm(all_scaled[good] - X_scaled, axis=1).argsort()[:k]
        return self.df_raw[good].iloc[idx][[
            "video title", "view count", "like count",
            "comment count", "length_sec", "tags", "video id"
        ]].to_dict(orient="records")

    # ─── Gemini + cache (unchanged) ---------------------------------------
    def _gemini(self, user, refs):
        prompt = (
            f"Candidate video:\n"
            f"• Title       : {user.get('title')}\n"
            f"• Description : {user.get('description')}\n"
            f"• Tags        : {user.get('tags')}\n"
            f"• Duration    : {user.get('duration_sec')} s\n\n"
            f"High-performing refs:\n{json.dumps(refs,indent=2)}\n\n"
            "Give bullet-point tips on:\n"
            "1. Title  2. Description  3. Tags  4. Ideal length  5. Thumbnail concept."
        )
        if prompt in self._cache:
            return {"tips": self._cache[prompt]}
        if not GEMINI_MODEL:
            return {"error": "Gemini not available"}

        delay, last_err = 1, None
        for _ in range(3):
            try:
                out  = GEMINI_MODEL.generate_content(prompt)
                tips = out.text.strip()
                self._cache[prompt] = tips
                self._save_cache()
                return {"tips": tips}
            except Exception as e:
                last_err = str(e)
                if "RATE_LIMIT" not in last_err and "quota" not in last_err:
                    break
                time.sleep(delay); delay *= 2
        return {"error": last_err or "Unknown Gemini error"}

    def _load_cache(self):
        if CACHE_FPATH.exists():
            try:
                return json.loads(CACHE_FPATH.read_text())
            except:
                pass
        return {}

    def _save_cache(self):
        try:
            CACHE_FPATH.write_text(json.dumps(self._cache, indent=2))
        except Exception as e:
            print("cache write failed:", e)

# CLI use
if __name__ == "__main__":
    VideoOptimiser()
    print("✅ optimiser trained & saved to", MODEL_FPATH)
