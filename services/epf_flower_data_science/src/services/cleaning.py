import pandas as pd

def preprocess_iris_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the Iris dataset for training.
    Steps:
    - Remove unnecessary columns.
    - Handle missing values.
    - Encode categorical columns.
    - (Optional) Normalize numeric features.
    """
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    
    df = df.dropna()

    if "Species" in df.columns:
        df["Species_Encoded"] = df["Species"].astype("category").cat.codes

    return df


from sklearn.model_selection import train_test_split
import pandas as pd

def split_iris_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split the Iris dataset into train and test sets.
    
    Args:
        df (pd.DataFrame): The processed dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        train_df (pd.DataFrame): Training set.
        test_df (pd.DataFrame): Test set.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df
