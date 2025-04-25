"""
ML-powered hospital recommender

Training target  (y)  = 2*distance_score + quality_score + experience_score + safety_score
Features (X)          = dist_mi, #cases, #adverse, risk_adj_rate, rating_num
Model                 = GradientBoostingRegressor
"""
import math, warnings, joblib, json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute  import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler


# ───────────────────────── CONSTANTS
EARTH_RADIUS_MI = 3958.8
CSV_FPATH       = Path("data/hospital_data.csv")
MODEL_FPATH     = Path("hospital_model.pkl")
COL_MAX_PATH    = Path("hospital_colmax.json")
FEATURES        = ["dist_mi", "# of cases", "# of adverse events",
                   "risk-adjusted rate", "rating_num"]

# ───────────────────────── HELPERS
def haversine(lat1, lon1, lat2, lon2):
    φ1, φ2 = map(math.radians, (lat1, lat2))
    Δφ, Δλ = map(math.radians, (lat2 - lat1, lon2 - lon1))
    a = (math.sin(Δφ/2)**2 +
         math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2)
    return EARTH_RADIUS_MI * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def rating_to_num(txt: str) -> float:
    if isinstance(txt, str):
        t = txt.lower().strip()
        if t.startswith("better"): return 10.0
        if t.startswith("worse"):  return 4.0
        if t.startswith("as"):     return 7.0
    return 5.0


# ───────────────────────── RECOMMENDER
class HospitalRecommender:
    def __init__(self, csv_path: str | Path = CSV_FPATH):
        self.csv_path = Path(csv_path)
        self.df_raw   = self._load_csv()
        self.model    = None
        self.scaler   = None
        self.imputer  = None
        self.col_max  = {}
        self._ensure_trained()

    # ---------- PUBLIC ----------
    def recommend(self, disease: str, user_lat: float, user_lon: float,
                  radius_miles: float = 50, top_n: int = 3) -> List[Dict]:

        df = self.df_raw[self.df_raw["performance measure"]
                         .str.casefold() == disease.casefold()].copy()
        if df.empty:
            return []

        df["dist_mi"] = df.apply(lambda r: haversine(user_lat, user_lon,
                                                    r["latitude"], r["longitude"]), axis=1)
        df = df[df["dist_mi"] <= radius_miles]
        if df.empty:
            return []
        df = df.drop_duplicates(subset=["hospital"])

        # ML features
        X = df[FEATURES].copy()
        X[np.isinf(X)] = np.nan
        X = self.imputer.transform(X)
        Xs = self.scaler.transform(X)
        df["predicted_score"] = self.model.predict(Xs)

        # ── recompute subscores (distance now scales to slider radius) ──
        dist_div = radius_miles or 1
        df["score_distance"]   = 5  * (1 - df["dist_mi"] / dist_div).clip(lower=0)
        df["score_experience"] = 10 * (df["# of cases"]          / self.col_max["# of cases"])
        df["score_safety"]     = 10 * (1 - df["# of adverse events"] / self.col_max["# of adverse events"])
        risk_score             = 10 * (1 - df["risk-adjusted rate"] / self.col_max["risk-adjusted rate"])
        df["score_quality"]    = (risk_score + df["rating_num"]) / 2

        df.fillna({"score_distance": 2.5, "score_experience": 5,
                   "score_safety": 5, "score_quality": 5}, inplace=True)

        df = df.sort_values("predicted_score", ascending=False).head(top_n)

        return [{
            "hospital"        : r["hospital"],
            "latitude"        : float(r["latitude"]),
            "longitude"       : float(r["longitude"]),
            "distance_mi"     : round(float(r["dist_mi"]), 2),

            # exact points for UI
            "score_distance"  : round(float(r["score_distance"]), 2),
            "score_quality"   : round(float(r["score_quality"]), 2),
            "score_experience": round(float(r["score_experience"]), 2),
            "score_safety"    : round(float(r["score_safety"]), 2),

            "predicted_score" : round(float(r["predicted_score"]), 2),
            "phone"           : r.get("phone", "")
        } for _, r in df.iterrows()]

    # ---------- INTERNAL ----------
    def _ensure_trained(self):
        if MODEL_FPATH.exists() and COL_MAX_PATH.exists():
            obj = joblib.load(MODEL_FPATH)
            self.model, self.scaler, self.imputer = obj["model"], obj["scaler"], obj["imputer"]
            self.col_max = json.loads(COL_MAX_PATH.read_text())
        else:
            self.train()

    def train(self):
        df = self.df_raw.copy()
        self._calc_col_max(df)

        lat_c, lon_c = df["latitude"].mean(), df["longitude"].mean()
        df["dist_mi"] = df.apply(lambda r: haversine(lat_c, lon_c,
                                                    r["latitude"], r["longitude"]), axis=1)

        df["distance_score"]   = 5  * (1 - df["dist_mi"] / self.col_max["dist_mi"])
        df["experience_score"] = 10 * (df["# of cases"] / self.col_max["# of cases"])
        df["safety_score"]     = 10 * (1 - df["# of adverse events"] / self.col_max["# of adverse events"])
        risk = 10 * (1 - df["risk-adjusted rate"] / self.col_max["risk-adjusted rate"])
        df["quality_score"]    = (risk + df["rating_num"]) / 2
        df.fillna({"distance_score": 2.5, "experience_score": 5,
                   "safety_score": 5, "quality_score": 5}, inplace=True)

        # <<-- WEIGHTED TARGET: distance is now doubled -->
        df["target_final"] = (
            2 * df["distance_score"]
            + df["quality_score"]
            + df["experience_score"]
            + df["safety_score"]
        )

        X, y = df[FEATURES], df["target_final"].values
        self.imputer = SimpleImputer(strategy="median").fit(X)
        X_imp = self.imputer.transform(X)
        self.scaler = StandardScaler().fit(X_imp)
        X_scaled = self.scaler.transform(X_imp)

        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        self.model = GradientBoostingRegressor(n_estimators=400,
                                               learning_rate=0.05,
                                               max_depth=3,
                                               random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_tr, y_tr)

        joblib.dump({"model": self.model, "scaler": self.scaler, "imputer": self.imputer}, MODEL_FPATH)
        COL_MAX_PATH.write_text(json.dumps(self.col_max))
        print("✔️ model trained  R²(test)=", round(self.model.score(X_te, y_te), 3))

    def _calc_col_max(self, df):
        self.col_max = {
            "dist_mi":             float(df.apply(lambda r: haversine(df["latitude"].mean(),
                                                                   df["longitude"].mean(),
                                                                   r["latitude"], r["longitude"]), axis=1).max() or 1.0),
            "# of cases":          float(df["# of cases"].max() or 1.0),
            "# of adverse events": float(df["# of adverse events"].max() or 1.0),
            "risk-adjusted rate":  float(df["risk-adjusted rate"].max() or 1.0)
        }

    def _load_csv(self):
        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.lower().str.strip()
        req = ["hospital", "performance measure", "latitude", "longitude",
               "# of cases", "# of adverse events", "risk-adjusted rate", "hospital ratings"]
        miss = [c for c in req if c not in df.columns]
        if miss:
            raise ValueError("CSV missing: " + ",".join(miss))
        num = ["latitude", "longitude", "# of cases", "# of adverse events", "risk-adjusted rate"]
        df[num] = df[num].apply(pd.to_numeric, errors="coerce", downcast="float")
        df.dropna(subset=["hospital", "performance measure", "latitude", "longitude"], inplace=True)
        df["rating_num"] = df["hospital ratings"].apply(rating_to_num)
        return df

if __name__ == "__main__":
    HospitalRecommender().train()
