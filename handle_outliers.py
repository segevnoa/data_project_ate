from sklearn.ensemble import IsolationForest
from handle_missing_values import *
import utilities


# outlier detection and removal using IQR method
def iqr_removal_method(df, attribute):
    q1 = df[attribute].quantile(0.25)
    q3 = df[attribute].quantile(0.75)
    iqr = q3 - q1

    low = max(0, q1 - 1.5 * iqr)
    high = q3 + 1.5 * iqr

    iqr_clean_df = df[(df[attribute] >= low) & (df[attribute] <= high)]
    removed = len(df) - len(iqr_clean_df)
    print(f"IQR method removed {removed} tuples from '{attribute}'")
    return iqr_clean_df


# outlier detection and removal using Isolation Forest
def isolation_forest_detection(df, target_attribute, feature_attributes, contamination=0.01, random_state=42):
    # target = the attribute that we want to clean
    # feature = list of attributes used for the detection
    # contamination = fraction of outliers expected (0.01 = 1%)
    # random_state = seed for reproducibility

    original_size = len(df)

    confounders_to_edlevel = [
        'Age',
        'Continent',
        'Ethnicity',
        'Gender'
    ]

    iso = IsolationForest(contamination=contamination, random_state=random_state)
    preds = iso.fit_predict(df[feature_attributes])

    df['__outlier__'] = preds
    valid_indices = df[df['__outlier__'] == 1].index

    df_to_return = df.drop(index=df.index.difference(valid_indices))

    # self check
    print(f"Isolation Forest removed {original_size - len(df_to_return)} rows as outliers from '"
          f"{target_attribute}' based on {feature_attributes}")

    return df_to_return


# outlier detection and removal using Standard Deviation (Z-score)
def standard_deviation_detection(df, target_attribute, threshold=3):
    mean = df[target_attribute].mean()
    std = df[target_attribute].std()

    low = mean - threshold * std
    high = mean + threshold * std

    mask = (df[target_attribute] >= low) & (df[target_attribute] <= high)
    valid_indices = df[mask].index

    df_cleaned = df.drop(index=df.index.difference(valid_indices))

    # self check
    print(f"Standard Deviation method removed {len(df) - len(df_cleaned)} outliers from '{target_attribute}'")

    return df_cleaned

