# Functions to check things in data
import pandas as pd
import numpy as np


def validate_correlation(df, threshold=0.95):
    assert(isinstance(df, pd.DataFrame))
    assert(len(df.columns) > 1), "Needed more than one column"    

    # Create correlation matrix
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f"Correlation is: {upper}")
    print(f"Correlation above threshold {threshold} for columns {to_drop}")