import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """Preprocesses the Titanic dataset by handling missing values, encoding categorical features, and scaling numerical columns."""
    
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df['Has_Cabin'] = df['Cabin'].notnull().astype(int)
    df.drop(columns=['Cabin'], inplace=True)

    df["Sex"] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

    return df