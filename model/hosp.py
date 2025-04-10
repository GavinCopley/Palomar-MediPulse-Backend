import pandas as pd
import os
import joblib
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def haversine(lat1, lon1, lat2, lon2):
    """
    Compute Haversine distance between two lat/longs in miles.
    """
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2)**2 
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

class HospMLModel:
    def __init__(self,
                 csv_path="model/hospitaldatamodified.csv",
                 model_path="hospital_model.pkl",
                 default_lat=32.7157,   # e.g., San Diego
                 default_lon=-117.1611):
        self.csv_path = csv_path
        self.model_path = model_path
        self.default_lat = default_lat
        self.default_lon = default_lon

        self.model = None
        self.disease_encoder = LabelEncoder()
        self.priority_encoder = LabelEncoder()
        self.scaler = None
        self.feature_columns = []

    def train_model(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.strip().str.lower()

        # Ensure essential columns
        required_cols = ["performance measure", "latitude", "longitude", "hospital",
                         "# of cases", "# of adverse events", "risk-adjusted rate"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' missing in CSV.")

        # Drop rows missing essential fields
        df = df.dropna(subset=["performance measure", "latitude", "longitude", "hospital"])

        # Compute distance from a default lat/lon (for training)
        df["distance"] = df.apply(
            lambda row: haversine(self.default_lat, self.default_lon, row["latitude"], row["longitude"]),
            axis=1
        )

        # Impute numeric columns
        numeric_cols = ["# of cases", "# of adverse events", "risk-adjusted rate"]
        imputer = SimpleImputer(strategy="mean")
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        # If your CSV doesn't have a 'priority' column, we create a dummy
        if "priority" not in df.columns:
            df["priority"] = "generic"

        # Encode disease & priority
        df["disease_encoded"] = self.disease_encoder.fit_transform(df["performance measure"])
        df["priority_encoded"] = self.priority_encoder.fit_transform(df["priority"])

        # Optionally weight them
        df["disease_encoded"] *= 2.0
        df["priority_encoded"] *= 4.0

        # Feature matrix
        X = df[[
            "disease_encoded",
            "priority_encoded",
            "# of cases",
            "# of adverse events",
            "risk-adjusted rate",
            "distance"
        ]]
        y = df["hospital"]

        self.feature_columns = X.columns.tolist()

        # Scale numeric features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=0.25,
            stratify=y,
            random_state=42
        )

        # Train RandomForest
        self.model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        self.model.fit(X_train, y_train)

        acc = self.model.score(X_test, y_test)
        print(f"✅ Model trained. Accuracy: {acc:.2f}")

        # Save everything
        joblib.dump({
            "model": self.model,
            "disease_encoder": self.disease_encoder,
            "priority_encoder": self.priority_encoder,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns
        }, self.model_path)

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        data = joblib.load(self.model_path)
        self.model = data["model"]
        self.disease_encoder = data["disease_encoder"]
        self.priority_encoder = data["priority_encoder"]
        self.scaler = data["scaler"]
        self.feature_columns = data["feature_columns"]

    def predict_ranked(self, disease, priority, max_distance, limit=3):
        """
        Return up to 'limit' hospitals ranked by the model's confidence.

        :param disease: string, e.g. "Acute Stroke"
        :param priority: string, e.g. "quality"
        :param max_distance: float, max miles user is willing to travel
        :param limit: int, how many hospitals to return
        :return: list of dicts: 
                 [
                   {
                     'hospital': ...,
                     'latitude': ...,
                     'longitude': ...,
                     'distance': ...,
                     'score': ...
                   },
                   ...
                 ]
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Encode user inputs
        try:
            disease_val = self.disease_encoder.transform([disease])[0] * 2.0
        except ValueError:
            disease_val = 0.0

        try:
            priority_val = self.priority_encoder.transform([priority])[0] * 2.0
        except ValueError:
            priority_val = 0.0

        # Load CSV
        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.strip().str.lower()
        df = df.dropna(subset=["performance measure", "latitude", "longitude", "hospital"])

        # Compute actual distance from user-lat/lon (defaults) to each hospital
        df["distance"] = df.apply(
            lambda row: haversine(self.default_lat, self.default_lon, row["latitude"], row["longitude"]),
            axis=1
        )

        # Filter by max_distance
        df = df[df["distance"] <= float(max_distance)]
        if df.empty:
            return []  # No hospitals within range

        # Filter by disease name
        mask = df["performance measure"].str.lower() == disease.lower()
        df = df[mask]
        if df.empty:
            return []

        # Impute numeric columns for the filtered set
        numeric_cols = ["# of cases", "# of adverse events", "risk-adjusted rate"]
        imputer = SimpleImputer(strategy="mean")
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        # For each row, set the encoded disease/priority
        df["disease_encoded"] = disease_val
        df["priority_encoded"] = priority_val

        # Build feature matrix in correct order
        X_candidate = df[[
            "disease_encoded",
            "priority_encoded",
            "# of cases",
            "# of adverse events",
            "risk-adjusted rate",
            "distance"
        ]]

        X_candidate_scaled = self.scaler.transform(X_candidate)

        # Probability predictions
        probs = self.model.predict_proba(X_candidate_scaled)
        hospital_classes = self.model.classes_

        # For each row, find the probability that the model identifies
        # that row's hospital as the predicted label:
        row_hospitals = df["hospital"].values

        results = []
        for i, hosp_name in enumerate(row_hospitals):
            # Probability that row i is "hosp_name"
            if hosp_name in hospital_classes:
                col_idx = list(hospital_classes).index(hosp_name)
                score = probs[i][col_idx]
            else:
                score = 0.0

            results.append({
                "hospital": hosp_name,
                "latitude": float(df.iloc[i]["latitude"]),
                "longitude": float(df.iloc[i]["longitude"]),
                "distance": float(df.iloc[i]["distance"]),
                "score": float(score)
            })

        # Sort results by score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)

        # Return up to 'limit'
        return results[:limit]

if __name__ == "__main__":
    model = HospMLModel()
    model.train_model()
