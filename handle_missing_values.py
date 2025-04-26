from sklearn.experimental import enable_iterative_imputer  # Required to activate
from sklearn.impute import IterativeImputer
import pandas as pd


# impute missing values with mean/method/mode
def impute_attribute(df, attribute, method):
    # imputed_df = df.copy()

    if method == 'mean':
        value = df[attribute].mean()
    elif method == 'median':
        value = df[attribute].median()
        print(value)
    elif method == 'mode':
        value = df[attribute].value_counts().idxmax()
    else:
        raise ValueError("Method must be 'mean', 'median', or 'mode'")

    df[attribute] = df[attribute].fillna(value)
    return df


# filling missing values using Iterative Impute
def iterative_impute(df, target_attribute, feature_attributes, random_state=42):
    data = df[feature_attributes + [target_attribute]]
    missing_before = data[target_attribute].isna().sum()

    imputer = IterativeImputer(random_state=random_state)
    imputed_array = imputer.fit_transform(data)

    # bringing the attribute after the imputation back into the dataset
    df[target_attribute] = imputed_array[:, -1]

    # self check
    print(
        f"✅ Imputed {missing_before} missing values in '{target_attribute}'"
        f" using IterativeImputer with features: {feature_attributes}")
    return df
