import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def split_data(df, target="Survived"):
    """Split the dataset into train and validation sets."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_logistic_model(X_train, y_train):
    """Train a logistic regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def get_model_coefficients(model, feature_names):
    """Return a sorted DataFrame of logistic regression coefficients."""
    coefficients = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", key=abs, ascending=False)
    return coefficients