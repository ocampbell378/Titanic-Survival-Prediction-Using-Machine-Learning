import sys
import os
# Ensure scripts/ folder is found when running this script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import scripts.load as load  # Now this works when running python scripts/eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def distribution_data(df):
    """Generates visualizations for survival, age, passenger class, and gender distributions."""

    if "Survived" in df.columns:
        plt.figure(figsize=(6, 4))
        survival_counts = df["Survived"].value_counts(normalize=True) * 100  
        ax = sns.barplot(x=survival_counts.index, y=survival_counts.values)
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}%', 
                        (p.get_x() + p.get_width() / 2, p.get_height() / 2), 
                        ha='center', va='bottom', fontsize=12, color='black')
        plt.title("Survival Percentage")
        plt.xlabel("Survived (0 = No, 1 = Yes)")
        plt.ylabel("Percentage (%)")
        plt.ylim(0, 100)
        plt.show()

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Age"].dropna(), bins=30, kde=True)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(6, 4))
    pclass_counts = df["Pclass"].value_counts(normalize=True) * 100  
    ax = sns.barplot(x=pclass_counts.index, y=pclass_counts.values)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2, p.get_height() / 2),  
                    ha='center', va='center', fontsize=12, color='black')
    plt.title("Passenger Class Distribution")
    plt.xlabel("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)
    plt.show()

    plt.figure(figsize=(6, 4))
    sex_counts = df["Sex"].value_counts(normalize=True) * 100  
    ax = sns.barplot(x=sex_counts.index, y=sex_counts.values)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2, p.get_height() / 2),  
                    ha='center', va='center', fontsize=12, color='black')
    plt.title("Gender Distribution")
    plt.xlabel("Gender")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)
    plt.show()


def plot_survival_correlation(df):
    """Plots a heatmap showing survival rate correlations with numerical features."""
    plt.figure(figsize=(12, 6))

    df = df.rename(columns={
        "SibSp": "Siblings/Spouses Aboard",
        "Parch": "Parents/Children Aboard",
        "Fare": "Passenger Fare"
    }).drop(columns=["PassengerId", "Name", "Age", "Pclass", "Embarked", "Sex"], errors="ignore")

    correlation = df.select_dtypes(include=["number"]).corr()["Survived"].drop("Survived")

    sns.heatmap(correlation.to_frame(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar_kws={'label': "Survival Correlation (>0 = Higher Survival, <0 = Lower Survival)"})
    plt.title("Survival Correlation with Numerical Features")
    plt.subplots_adjust(left=0.3, right=0.85)
    plt.show()


def plot_categorical_survival(df):
    """Plots survival rates for Pclass, Embarked, and Sex using countplots."""

    plt.figure(figsize=(6, 4))
    sns.countplot(x="Pclass", hue="Survived", data=df)
    plt.title("Survival Count by Passenger Class")
    plt.xlabel("Passenger Class (1st, 2nd, 3rd)")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.countplot(x="Embarked", hue="Survived", data=df)
    plt.title("Survival Count by Embarkation Point")
    plt.xlabel("Embarkation Port (C = Cherbourg, Q = Queenstown, S = Southampton)")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.countplot(x="Sex", hue="Survived", data=df)
    plt.title("Survival Count by Sex")
    plt.xlabel("Sex")
    plt.ylabel("Count")
    plt.show()


def plot_survival_by_age(df):
    """Plots a heatmap of survival rates across binned age groups."""
    
    df = df.copy()
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80])

    survival_rates = df.groupby("AgeBin", observed=False)["Survived"].mean().to_frame().T

    plt.figure(figsize=(10, 2))
    sns.heatmap(survival_rates, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5,
                cbar_kws={'label': 'Survival Rate'})
    plt.title("Survival Rate by Age Group")
    plt.xlabel("Age Group")
    plt.yticks([], [])
    plt.show()


if __name__ == "__main__":
    train_df, test_df = load.load_data()
    distribution_data(train_df)
    plot_categorical_survival(train_df)
    plot_survival_correlation(train_df)
    plot_survival_by_age(train_df)