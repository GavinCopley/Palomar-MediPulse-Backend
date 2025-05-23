"""
Video Optimiser — full-feature, NaN-safe (2025-05-21)
─────────────────────────────────────────────────────
Target (y)  = 0.6·view_rate + 0.3·like_rate + 0.1·comment_rate
Numeric     = length_sec, title_len, desc_len, tag_count, is_hd,
              has_captions, category_id, publish_dow, publish_hour
Text        = 2 modes
              • USE_EMBEDDINGS=1 → Sentence-Transformer → SVD
              • USE_EMBEDDINGS=0 → TF-IDF              → SVD
Model       = [Numeric ‖ Text] → GradientBoostingRegressor
Endpoints   = VideoOptimiser.suggest(payload)  (used by /api/optimize)
"""

from __future__ import annotations
import os, json, glob, warnings, time
from pathlib import Path

# ─── tip category helpers ─────────────────────────────
TIP_CATEGORIES = ["title", "description", "tags", "length", "thumbnail"]

def _empty_tip_block():
    """Return {examples:[], suggestions:[]} for every category."""
    return {c: {"examples": [], "suggestions": []} for c in TIP_CATEGORIES}

# ─── load .env so GEMINI vars become visible ─────────────────────────────
try:
    from dotenv import load_dotenv          # pip install python-dotenv
    load_dotenv("Gemini_optimize.env")      # contains GEMINI_API_KEY, USE_GEMINI
except ModuleNotFoundError:
    pass

# ╭──────────────────────────────────────────────────────────────────────╮
# │ Config & constants                                                  │
# ╰──────────────────────────────────────────────────────────────────────╯
DATA_DIR        = Path("data/yt")
MODEL_FPATH     = Path("video_model.pkl")
CACHE_FPATH     = Path("data/gemini_cache.json")
CACHE_FPATH.parent.mkdir(parents=True, exist_ok=True)

IMPROVEMENT_FACTOR   = float(os.getenv("IMPROVEMENT_FACTOR", "0.10"))

USE_EMBEDDINGS       = os.getenv("USE_EMBEDDINGS", "0") == "1"
EMBED_MODEL_NAME     = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
MAX_TFIDF_FEATURES   = int(os.getenv("MAX_TFIDF_FEATURES", "500"))
TEXT_SVD_DIM         = int(os.getenv("TEXT_SVD_DIM", "50"))

# ╭──────────────────────────────────────────────────────────────────────╮
# │ Libraries                                                           │
# ╰──────────────────────────────────────────────────────────────────────╯
import joblib, pandas as pd, numpy as np
from sklearn.ensemble      import GradientBoostingRegressor
from sklearn.impute        import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

if USE_EMBEDDINGS:
    try:
        from sentence_transformers import SentenceTransformer   # noqa: E402
    except ImportError:
        print("⚠️  sentence-transformers missing → falling back to TF-IDF.")
        USE_EMBEDDINGS = False

if not USE_EMBEDDINGS:
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

# ╭──────────────────────────────────────────────────────────────────────╮
# │ Optional Gemini setup                                              │
# ╰──────────────────────────────────────────────────────────────────────╯
# default = “1” whenever GEMINI_API_KEY exists
USE_GEMINI = (
    os.getenv("USE_GEMINI", "1" if os.getenv("GEMINI_API_KEY") else "0") == "1"
)
GEMINI_MODEL = None
if USE_GEMINI and (key := os.getenv("GEMINI_API_KEY", "").strip()):
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        GEMINI_MODEL = genai.GenerativeModel(
            os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
        )
    except Exception as e:
        print("⚠️  Gemini init failed:", e)
        USE_GEMINI = False

# ╭──────────────────────────────────────────────────────────────────────╮
# │ CSV helpers                                                        │
# ╰──────────────────────────────────────────────────────────────────────╯
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
    dfs=[]
    for fp in glob.glob(str(folder/"**/*.csv"), recursive=True):
        try:
            df=pd.read_csv(fp)
            df.columns=df.columns.str.lower().str.strip()
            dfs.append(df); print("✔︎ read", fp)
        except Exception as e:
            print("✗ skip", fp, "→", e)
    if not dfs:
        raise RuntimeError(f"No CSVs found in {folder}")
    return pd.concat(dfs, ignore_index=True)

def _prep_dataframe(df:pd.DataFrame)->pd.DataFrame:
    for old,new in RENAME.items():
        if old in df.columns: df.rename(columns={old:new}, inplace=True)
    for col,default in BASE_DEFAULTS.items():
        if col not in df.columns: df[col]=default

    df["length_sec"] = pd.to_numeric(df["duration sec"], errors="coerce")
    df["title_len"]  = df["video title"].astype(str).str.len()
    df["desc_len"]   = df["video description"].astype(str).str.len()
    df["tag_count"]  = df["tags"].fillna("").astype(str).str.split("|").str.len()
    df["is_hd"]      = df["definition"].str.contains("hd",case=False,na=False).astype(int)
    df["has_captions"]=df["captions"].astype(str).str.startswith(
                        ("t","y","1","T","Y")).astype(int)

    dt=pd.to_datetime(df["date published"], errors="coerce")
    df["publish_dow"] = dt.dt.dayofweek.fillna(0).astype(int)
    df["publish_hour"]= dt.dt.hour.fillna(12).astype(int)

    for c in ["view count","like count","comment count"]:
        df[c]=pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["view count","like count","comment count","length_sec"], inplace=True)

    df["view_rate"]    = df["view count"]/1000
    df["like_rate"]    = df["like count"]/1000
    df["comment_rate"] = df["comment count"]/1000
    return df

NUMERICAL_FEATS = [
    "length_sec","title_len","desc_len","tag_count",
    "is_hd","has_captions","category_id","publish_dow","publish_hour"
]

# ╭──────────────────────────────────────────────────────────────────────╮
# │ Optimiser class                                                     │
# ╰──────────────────────────────────────────────────────────────────────╯
class VideoOptimiser:
    def __init__(self,data_dir:Path=DATA_DIR):
        self.df_raw = _prep_dataframe(_load_all_csvs(data_dir))
        self.df_raw["engagement_score"]= (
            0.6*self.df_raw["view_rate"] +
            0.3*self.df_raw["like_rate"] +
            0.1*self.df_raw["comment_rate"]
        )
        self._ensure_trained()
        self._cache=self._load_cache()

    # ── public ----------------------------------------------------------
    def suggest(self,payload:dict,top_n:int=5)->dict:
        title = str(payload.get("title","") or "")
        desc  = str(payload.get("description","") or "")
        X_num = self._numeric_df(payload).values.astype(float)        # (1×9)
        X_txt = self._encode([f"{title} {desc}"])
        X     = np.hstack([np.atleast_2d(X_num), X_txt])              # (1×(9+text))
        Xp    = self.scaler.transform(self.imputer.transform(X))

        pred  = float(self.model.predict(Xp)[0])
        refs  = self._nearest(Xp, top_n)
        tips  = self._gemini(payload, refs) if USE_GEMINI else {"info":"Gemini disabled"}

        return {"predicted_engagement":round(pred,2),
                "improved_engagement":round(pred*(1+IMPROVEMENT_FACTOR),2),
                "reference_videos":refs,
                "gemini_tips":tips}

    # ── training / loading ---------------------------------------------
    def _ensure_trained(self):
        if MODEL_FPATH.exists():
            obj = joblib.load(MODEL_FPATH)
            self.model   = obj["model"]
            self.imputer = obj["imputer"]
            self.scaler  = obj["scaler"]
            self.tfidf   = obj["tfidf"]
            self.svd     = obj["svd"]
            self.embedder= obj.get("embedder")  # may be None
        else:
            self._train()

    def _train(self):
        X_num = self.df_raw[NUMERICAL_FEATS].values
        texts = (self.df_raw["video title"].fillna("")+" "+
                 self.df_raw["video description"].fillna("")).tolist()
        X_txt = self._encode(texts)
        X_all = np.hstack([X_num, X_txt])
        y     = self.df_raw["engagement_score"].values

        self.imputer=SimpleImputer(strategy="median").fit(X_all)
        X_imp=self.imputer.transform(X_all)
        self.scaler=StandardScaler().fit(X_imp)
        X_scaled=self.scaler.transform(X_imp)

        Xtr,Xte,ytr,yte=train_test_split(X_scaled,y,test_size=0.25,
                                         random_state=42)
        self.model=GradientBoostingRegressor(n_estimators=400,
                    learning_rate=0.05,max_depth=3,random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(Xtr,ytr)

        joblib.dump({"model":self.model,"imputer":self.imputer,"scaler":self.scaler,
                     "tfidf":getattr(self,"tfidf",None),
                     "svd":self.svd,
                     "embedder":getattr(self,"embedder",None)},
                    MODEL_FPATH)
        print("✅ model trained & saved ➜", MODEL_FPATH)

    # ── text encoding ---------------------------------------------------
    def _encode(self,texts:list[str])->np.ndarray:
        clean=["" if pd.isna(t) else str(t) for t in texts]   # NaN-safe
        if USE_EMBEDDINGS:
            if not hasattr(self,"embedder"):
                self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
            emb = self.embedder.encode(clean, show_progress_bar=False)
            emb = np.atleast_2d(emb)
            if not hasattr(self,"svd"):
                self.svd = TruncatedSVD(n_components=TEXT_SVD_DIM,
                                        random_state=42).fit(emb)
            return self.svd.transform(emb)

        # TF-IDF pathway
        if not hasattr(self,"tfidf"):
            self.tfidf = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES).fit(clean)
            self.svd   = TruncatedSVD(n_components=TEXT_SVD_DIM,
                                      random_state=42).fit(
                        self.tfidf.transform(clean))
        dense=self.svd.transform(self.tfidf.transform(clean))
        return np.atleast_2d(dense)

    # ── helpers ---------------------------------------------------------
    def _numeric_df(self,d:dict)->pd.DataFrame:
        return pd.DataFrame([{
            "length_sec"  : d.get("duration_sec"),
            "title_len"   : len(str(d.get("title",""))),
            "desc_len"    : len(str(d.get("description",""))),
            "tag_count"   : len(str(d.get("tags","")).split("|"))
                            if d.get("tags") else 0,
            "is_hd"       : int(bool(d.get("is_hd",1))),
            "has_captions": int(bool(d.get("has_captions",0))),
            "category_id" : d.get("category_id",0),
            "publish_dow" : d.get("publish_dow",0),
            "publish_hour": d.get("publish_hour",12)
        }])

    def _nearest(self,X_scaled,k:int):
        all_scaled=self.scaler.transform(
            self.imputer.transform(
                np.hstack([self.df_raw[NUMERICAL_FEATS].values,
                           self._encode((self.df_raw["video title"]+" "+
                                         self.df_raw["video description"]).tolist())])))
        good=self.df_raw["engagement_score"] > np.percentile(
              self.df_raw["engagement_score"],80)
        idx=np.linalg.norm(all_scaled[good]-X_scaled,axis=1).argsort()[:k]
        return self.df_raw[good].iloc[idx][[
            "video title","view count","like count","comment count",
            "length_sec","tags","video id"]].to_dict(orient="records")

    # ── gemini + cache --------------------------------------------------
    def _gemini(self, user: dict, refs: list[dict]) -> dict:
        """
        Calls Gemini and returns a dict:
          {
            "title":      { "examples": [...3], "suggestions":[...2] },
            "description":{ "examples": [...3], "suggestions":[...2] },
            …
          }
        If Gemini is unavailable, falls back to _empty_tip_block().
        """
        # 1) if already cached
        cache_key = json.dumps({"u": user, "r": refs}, sort_keys=True)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 2) bail out if Gemini not active
        if not GEMINI_MODEL:
            return _empty_tip_block()

        # 3) prompt – tell Gemini to answer in strict JSON
        prompt = f"""
You are an expert YouTube content strategist.

For the candidate video below **only** answer with valid JSON (no markdown)
following exactly this schema:

{{
  "title":       {{"examples": ["…", "…", "…"], "suggestions": ["…", "…"]}},
  "description": {{…}},
  "tags":        {{…}},
  "length":      {{…}},
  "thumbnail":   {{…}}
}}

Each category = 6 concrete EXAMPLES your client could copy-paste 
                + 3 short SUGGESTIONS / explanations. be specific with the explanations to the prompt. Do not make them generic, make thewm unique to the prompt. also include if it doesnt adhere to hospital guidelines/topic.

Candidate video:
- Title: {user.get('title')}
- Description: {user.get('description')}
- Tags: {user.get('tags')}
- Duration: {user.get('duration_sec')} s 

High-performing reference videos (trimmed):
{json.dumps([{k:v for k,v in r.items() if k in ("video title","tags","length_sec")} for r in refs][:3], indent=2)}
"""

        # 4) call Gemini (retry up to 3× on quota)
        delay, err = 1, None
        for _ in range(3):
            try:
                out = GEMINI_MODEL.generate_content(prompt)
                import re, textwrap

                raw = out.text.strip()

                # ── 1) remove ``` fences if Gemini wrapped the JSON in markdown
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.S).strip()

                # ── 2) keep only the first {...} block (guards against pre-ambles / epilogues)
                first, last = raw.find("{"), raw.rfind("}")
                if first != -1 and last != -1:
                    raw = raw[first : last + 1]

                # ── 3) now load
                tips = json.loads(raw)
                # minimal validation
                for cat in TIP_CATEGORIES:
                    tips.setdefault(cat, {"examples": [], "suggestions": []})
                self._cache[cache_key] = tips
                self._save_cache()
                return tips
            except Exception as e:
                err = str(e)
                if "quota" not in err.lower() and "rate" not in err.lower():
                    break
                time.sleep(delay); delay *= 2

        # 5) failure → return empty structure
        print("Gemini error:", err)
        return _empty_tip_block()

    # ── cache helpers ---------------------------------------------------
    def _load_cache(self):
        if CACHE_FPATH.exists():
            try: return json.loads(CACHE_FPATH.read_text())
            except: pass
        return {}
    def _save_cache(self):
        try: CACHE_FPATH.write_text(json.dumps(self._cache,indent=2))
        except Exception as e:
            print("cache write failed:",e)

# ╭──────────────────────────────────────────────────────────────────────╮
# │ CLI – `python -m model.optimize`                                    │
# ╰──────────────────────────────────────────────────────────────────────╯
if __name__=="__main__":
    VideoOptimiser()
