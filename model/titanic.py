"""
File: titanic.py

Purpose:
- Contains a TitanicModel class to encapsulate data loading, cleaning, training, and prediction.
- Includes initTitanic() and testTitanic() helper methods to initialize and test the model.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import seaborn as sns

class TitanicModel:
    """A class used to represent the Titanic Model for passenger survival prediction.
    """
    _instance = None  # Singleton instance
    
    def __init__(self):
        """Constructor for TitanicModel. 
        Loads the data, but training is done in _train().
        """
        # Titanic logistic regression model
        self.model = None
        # Titanic decision tree (used for feature_importances_)
        self.dt = None
        # Common features used for the logistic regression
        self.features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'alone']
        # The "survived" column is the target 
        self.target = 'survived'
        # Load the raw Titanic dataset
        self.titanic_data = sns.load_dataset('titanic')
        # One-hot encoder for embarked columns
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    def _clean(self):
        """Private method to clean and prepare the Titanic dataset for training."""
        # Drop columns that won't be used
        self.titanic_data.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], 
                               axis=1, inplace=True)

        # Convert sex from 'male'/'female' to 1/0
        self.titanic_data['sex'] = self.titanic_data['sex'].apply(lambda x: 1 if x == 'male' else 0)

        # Convert alone from boolean True/False to 1/0
        self.titanic_data['alone'] = self.titanic_data['alone'].apply(lambda x: 1 if x == True else 0)

        # Drop rows where 'embarked' is NaN before encoding
        self.titanic_data.dropna(subset=['embarked'], inplace=True)

        # One-hot encode 'embarked' 
        onehot = self.encoder.fit_transform(self.titanic_data[['embarked']]).toarray()
        cols = ['embarked_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.titanic_data = pd.concat([self.titanic_data, onehot_df], axis=1)

        # Remove the original 'embarked' column
        self.titanic_data.drop(['embarked'], axis=1, inplace=True)

        # Extend features with the new one-hot columns
        self.features.extend(cols)

        # Drop remaining rows with any NaN
        self.titanic_data.dropna(inplace=True)

    def _train(self):
        """Private method to train both Logistic Regression and Decision Tree on the cleaned Titanic data."""
        # Prepare the data
        X = self.titanic_data[self.features]
        y = self.titanic_data[self.target]

        # Train logistic regression
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)

        # Train a decision tree
        self.dt = DecisionTreeClassifier()
        self.dt.fit(X, y)

    @classmethod
    def get_instance(cls):
        """Singleton access method to retrieve the trained TitanicModel."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._clean()
            cls._instance._train()
        return cls._instance

    def predict(self, passenger):
        """Predict the survival probability of a passenger.

        Args:
            passenger (dict): Keys must be:
                'name' (str), 
                'pclass' (int), 
                'sex' (str => 'male' or 'female'), 
                'age' (numeric), 
                'sibsp' (int),
                'parch' (int), 
                'fare' (numeric), 
                'embarked' (str => 'C','Q','S'),
                'alone' (bool).

        Returns:
            dict: probabilities {'die': float, 'survive': float}
        """
        # Convert passenger to a DataFrame
        passenger_df = pd.DataFrame(passenger, index=[0])

        # Convert sex to numeric
        passenger_df['sex'] = passenger_df['sex'].apply(lambda x: 1 if x == 'male' else 0)
        # Convert alone to numeric
        passenger_df['alone'] = passenger_df['alone'].apply(lambda x: 1 if x == True else 0)

        # One-hot encode 'embarked'
        onehot = self.encoder.transform(passenger_df[['embarked']]).toarray()
        cols = ['embarked_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)

        # Attach the new columns
        passenger_df = pd.concat([passenger_df, onehot_df], axis=1)

        # Drop columns no longer needed
        passenger_df.drop(['embarked', 'name'], axis=1, inplace=True)

        # Ensure passenger_df has all columns in the same order as self.features
        # Fill missing columns with 0 if necessary (in case they didn't specify a code path)
        for col in self.features:
            if col not in passenger_df.columns:
                passenger_df[col] = 0

        passenger_df = passenger_df[self.features]

        # Predict probabilities [prob_die, prob_survive]
        prob_die, prob_survive = np.squeeze(self.model.predict_proba(passenger_df))
        return {'die': prob_die, 'survive': prob_survive}

    def feature_weights(self):
        """Return DecisionTree feature importances as a dictionary."""
        importances = self.dt.feature_importances_
        return {feature: importance for feature, importance in zip(self.features, importances)}

def initTitanic():
    """Initialize the Titanic Model (loads data, cleans, trains)."""
    TitanicModel.get_instance()

def testTitanic():
    """Test method to ensure TitanicModel is up and running properly."""
    print("=== Titanic Model Test ===")
    
    # Sample passenger
    passenger = {
        'name': ['John Mortensen'],
        'pclass': 2,
        'sex': 'male',
        'age': 65,
        'sibsp': 1,
        'parch': 1,
        'fare': 16.00,
        'embarked': 'S',
        'alone': False
    }
    
    # Retrieve the TitanicModel
    model = TitanicModel.get_instance()
    # Predict
    probabilities = model.predict(passenger)
    print(f"Passenger: {passenger['name'][0]}")
    print(f" -> Probability of Death  : {probabilities.get('die'):.2%}")
    print(f" -> Probability of Survival : {probabilities.get('survive'):.2%}")

    # Feature importances
    print("\nFeature Importances (Decision Tree):")
    for feat, imp in model.feature_weights().items():
        print(f"  {feat}: {imp:.3f}")

# If you want to run this file directly, you can un-comment the following:
# if __name__ == "__main__":
#     testTitanic()
