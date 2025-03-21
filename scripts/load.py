import pandas as pd


def load_data(train_path="data/train.csv", test_path="data/test.csv"):
    """Load the Titanic dataset from CSV files."""

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def inspect_data(df, name="Dataset"):
    """Prints basic information about the dataset."""
    print(f"\n Inspecting {name}...\n")

    print("🔹 First 5 rows:\n")
    print(df.head().to_string())

    print("\n🔹 Dataset Info:\n")
    df.info()

    print("\n🔹 Summary Statistics:\n")
    print(df.describe().to_string())

    print("\n🔹 Missing Values:\n")
    print(df.isnull().sum().to_string())
    print("\n")


#Script testing
if __name__ == "__main__":
    train, test = load_data()
    inspect_data(train, "Train Data")
    inspect_data(test, "Test Data")