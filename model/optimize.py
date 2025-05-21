"""
ML-powered video optimiser
──────────────────────────
Target (y)  = 0.6*view_rate + 0.3*like_rate + 0.1*comment_rate
Features    = length_sec, title_len, desc_len, tag_count, is_hd,
              has_captions, category_id, publish_dow, publish_hour
Model       = GradientBoostingRegressor
LLM tips    = Google Gemini-Pro (optional)
"""

from __future__ import annotations
import os, json, glob, warnings, time
from pathlib import Path

# ─── explicit load of your custom env file ─────────────────────────────────
try:
    from dotenv import load_dotenv            # pip install python-dotenv
    load_dotenv("Gemini_optimize.env")         # read your Gemini_optimize.env
except ModuleNotFoundError:
    pass                                       # fallback if python-dotenv isn’t installed

import joblib, pandas as pd, numpy as np
from sklearn.ensemble   import GradientBoostingRegressor
from sklearn.impute     import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ───────────────────────── paths & constants
DATA_DIR        = Path("data/yt")
MODEL_FPATH     = Path("video_model.pkl")
CACHE_FPATH     = Path("data/gemini_cache.json")
CACHE_FPATH.parent.mkdir(parents=True, exist_ok=True)

# Configurable uplift factor (e.g. 0.1 = 10% improvement)
IMPROVEMENT_FACTOR = float(os.getenv("IMPROVEMENT_FACTOR", "0.1"))

NUMERICAL_FEATS = [
    "length_sec","title_len","desc_len","tag_count",
    "is_hd","has_captions","category_id","publish_dow","publish_hour"
]

# ───────────────────────── Gemini set-up
USE_GEMINI   = os.getenv("USE_GEMINI", "1") != "0"
GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = None
if USE_GEMINI and GEMINI_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        GEMINI_MODEL = genai.GenerativeModel(
            os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
        )
    except Exception as e:
        print("⚠️  Gemini unavailable:", e)
        USE_GEMINI = False

# ───────────────────────── CSV helpers
RENAME = {
    "datepublished":"date published","date_published":"date published",
    "videoid":"video id","video_id":"video id",
    "videotitle":"video title","video_title":"video title",
    "videodescription":"video description","video_description":"video description",
    "durationsec":"duration sec","duration_sec":"duration sec",
    "viewcount":"view count","view_count":"view count",
    "likecount":"like count","like_count":"like count",
    "commentcount":"comment count","comment_count":"comment count",
}
BASE_DEFAULTS = {
    "date published":pd.NaT,"video id":"","video title":"","video description":"",
    "tags":"","duration sec":0,"view count":0,"like count":0,"comment count":0,
    "definition":"","captions":"F","category_id":0
}

def _load_all_csvs(folder:Path)->pd.DataFrame:
    dfs=[]
    for fp in glob.glob(str(folder/"**/*.csv"), recursive=True):
        try:
            df=pd.read_csv(fp)
            df.columns=df.columns.str.lower().str.strip()
            dfs.append(df)
            print("✔︎ read", fp)
        except Exception as e:
            print("✗ skip", fp, "→", e)
    if not dfs: raise RuntimeError(f"❌ No CSVs found in {folder}")
    return pd.concat(dfs, ignore_index=True)

def _prep_dataframe(df:pd.DataFrame)->pd.DataFrame:
    for old,new in RENAME.items():
        if old in df.columns: df.rename(columns={old:new}, inplace=True)
    for col,default in BASE_DEFAULTS.items():
        if col not in df.columns:
            df[col]=default

    df["length_sec"]=pd.to_numeric(df["duration sec"], errors="coerce")
    df["title_len"]=df["video title"].astype(str).str.len()
    df["desc_len"]=df["video description"].astype(str).str.len()
    df["tag_count"]=df["tags"].fillna("").astype(str).str.split("|").str.len()
    df["is_hd"]=df["definition"].str.contains("hd",case=False,na=False).astype(int)
    df["has_captions"]=df["captions"].astype(str).str.startswith(("t","y","1","T","Y")).astype(int)

    dt=pd.to_datetime(df["date published"], errors="coerce")
    df["publish_dow"]=dt.dt.dayofweek.fillna(0).astype(int)
    df["publish_hour"]=dt.dt.hour.fillna(12).astype(int)

    for c in ["view count","like count","comment count"]:
        df[c]=pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["view count","like count","comment count","length_sec"], inplace=True)

    df["view_rate"]=df["view count"]/1000
    df["like_rate"]=df["like count"]/1000
    df["comment_rate"]=df["comment count"]/1000
    return df

# ───────────────────────── optimiser
class VideoOptimiser:
    def __init__(self, data_dir:Path=DATA_DIR):
        self.df_raw=_prep_dataframe(_load_all_csvs(data_dir))
        self.df_raw["engagement_score"] = (
            0.6*self.df_raw["view_rate"] +
            0.3*self.df_raw["like_rate"] +
            0.1*self.df_raw["comment_rate"]
        )
        self._ensure_trained()
        self._cache = self._load_cache()

    def suggest(self, vd:dict, top_n:int=5)->dict:
        # predict
        X = self.scaler.transform(self.imputer.transform(self._dict_df(vd)[NUMERICAL_FEATS]))
        pred = float(self.model.predict(X)[0])
        
        # Cap the prediction at 100
        pred = min(pred, 100.0)
        
        # simulate improved score
        improved = round(pred * (1 + IMPROVEMENT_FACTOR), 2)
        # Also cap the improved score
        improved = min(improved, 100.0)

        # nearest refs and tips
        refs = self._nearest(X, top_n)
        tips = self._gemini(vd, refs) if USE_GEMINI else {"error":"Gemini disabled"}

        return {
            "predicted_engagement": round(pred,2),
            "improved_engagement": improved,
            "reference_videos": refs,
            "gemini_tips": tips
        }

    def _ensure_trained(self):
        if MODEL_FPATH.exists():
            obj=joblib.load(MODEL_FPATH)
            self.model, self.imputer, self.scaler = obj["model"], obj["imputer"], obj["scaler"]
        else:
            self._train()

    def _train(self):
        X = self.df_raw[NUMERICAL_FEATS]
        y = self.df_raw["engagement_score"].values
        self.imputer = SimpleImputer(strategy="median").fit(X)
        Xs = self.scaler.fit_transform(self.imputer.transform(X))
        Xtr,Xte,ytr,yte = train_test_split(Xs,y,test_size=0.25,random_state=42)
        self.model = GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3, random_state=42
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(Xtr,ytr)
        joblib.dump({
            "model":self.model,
            "imputer":self.imputer,
            "scaler":self.scaler
        }, MODEL_FPATH)

    def _dict_df(self, d:dict)->pd.DataFrame:
        return pd.DataFrame([{
            "length_sec":d.get("duration_sec"),
            "title_len":len(d.get("title","")),
            "desc_len":len(d.get("description","")),
            "tag_count":len(d.get("tags","").split("|")) if d.get("tags") else 0,
            "is_hd":int(bool(d.get("is_hd",1))),
            "has_captions":int(bool(d.get("has_captions",0))),
            "category_id":d.get("category_id",0),
            "publish_dow":d.get("publish_dow",0),
            "publish_hour":d.get("publish_hour",12)
        }])

    def _nearest(self, X, npicks):
        all_scaled = self.scaler.transform(self.imputer.transform(self.df_raw[NUMERICAL_FEATS]))
        mask = self.df_raw["engagement_score"] > np.percentile(self.df_raw["engagement_score"],80)
        idx = np.linalg.norm(all_scaled[mask]-X,axis=1).argsort()[:npicks]
        return self.df_raw[mask].iloc[idx][
            ["video title","view count","like count","comment count","length_sec","tags","video id"]
        ].to_dict(orient="records")

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
            return {"error":"Gemini not available"}

        delay, last_err = 1, None
        for _ in range(3):
            try:
                out = GEMINI_MODEL.generate_content(prompt)
                tips = out.text.strip()
                self._cache[prompt] = tips
                self._save_cache()
                return {"tips": tips}
            except Exception as e:
                last_err = str(e)
                if "RATE_LIMIT" not in last_err and "quota" not in last_err:
                    break
                time.sleep(delay)
                delay *= 2
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
            CACHE_FPATH.write_text(json.dumps(self._cache,indent=2))
        except Exception as e:
            print("cache write failed:", e)

if __name__=="__main__":
    VideoOptimiser()
