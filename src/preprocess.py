# src/data_prep.py

import numpy as np
import pandas as pd

def load_data(npz_path: str = "notebook/preprocessed_data.npz"):
   
    data = np.load(npz_path)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    return X_train, y_train, X_test, y_test


# def train_test_split_and_balance(df_final: pd.DataFrame,
#                                  train_ratio: float = 0.8,
#                                  random_state: int = 42):
   
#     np.random.seed(random_state)
#     indices = np.random.permutation(len(df_final))

#     train_size = int(train_ratio * len(df_final))
#     train_idx = indices[:train_size]
#     test_idx = indices[train_size:]

#     train_df = df_final.iloc[train_idx].reset_index(drop=True)
#     test_df  = df_final.iloc[test_idx].reset_index(drop=True)

#     # Undersampling EXACTLY like notebook
#     fraud_train  = train_df[train_df["Class"] == 1]
#     normal_train = train_df[train_df["Class"] == 0]

#     normal_train_undersampled = normal_train.sample(
#         n=len(fraud_train),
#         random_state=random_state
#     )

#     balanced_train_df = pd.concat([fraud_train, normal_train_undersampled])
#     balanced_train_df = balanced_train_df.sample(
#         frac=1, random_state=random_state
#     ).reset_index(drop=True)

#     # Create X/y
#     X_train = balanced_train_df.drop("Class", axis=1).values
#     y_train = balanced_train_df["Class"].values

#     X_test = test_df.drop("Class", axis=1).values
#     y_test = test_df["Class"].values

#     feature_names = balanced_train_df.drop("Class", axis=1).columns.tolist()

#     return X_train, y_train, X_test, y_test, feature_names
