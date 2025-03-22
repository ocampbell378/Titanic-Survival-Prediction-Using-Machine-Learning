def engineer_features(df):
    """Adds new features and drops irrelevant columns to enhance model performance."""

    df["FamilySize"] = df["SibSp"] + df["Parch"]
    
    df.drop(columns=["Name", "Ticket"], inplace=True)
    return df