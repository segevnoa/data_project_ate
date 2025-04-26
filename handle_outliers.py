from sklearn.ensemble import IsolationForest
from handle_missing_values import *
import utilities


# outlier detection and removal using IQR method
def iqr_removal_method(df, attribute):
    q1 = df[attribute].quantile(0.25)
    q3 = df[attribute].quantile(0.25)
    iqr = q3 - q1

    low = max(0, q1 - 1.5 * iqr)
    high = q3 + 1.5 * iqr

    iqr_clean_df = df[(df[attribute] >= low) & (df[attribute] <= high)]
    return iqr_clean_df


# outlier detection and removal using Isolation Forest
def isolation_forest_detection(df, target_attribute, feature_attributes, contamination=0.01, random_state=42):
    # target = the attribute that we want to clean
    # feature = list of attributes used for the detection
    # contamination = fraction of outliers expected (0.01 = 1%)
    # random_state = seed for reproducibility
    # df_copy = df.dropna(subset=feature_attributes + [target_attribute]).copy()

    # df = impute_attribute(df, 'ConvertedCompYearly', 'mean')
    # df = impute_attribute(df, 'EdLevel', 'mode')

    # df = impute_attribute(df, 'ConvertedCompYearly', 'median')
    # df = impute_attribute(df, 'EdLevel', 'mode')
    # df = df.dropna() # for other values

    confounders_to_edlevel = [
        'Age',
        'Continent',
        'Ethnicity',
        'Gender'
    ]

    df = utilities.encode_demographics(df)
    df['EdLevel'] = df['EdLevel'].apply(utilities.bin_education_level)
    df['EdLevel'] = df['EdLevel'].map({'Low': 0, 'High': 1})
    df = iterative_impute(df, 'ConvertedCompYearly', confounders_to_edlevel)
    df = iterative_impute(df, 'EdLevel', confounders_to_edlevel)

    df = df.dropna()

    iso = IsolationForest(contamination=contamination, random_state=random_state)
    preds = iso.fit_predict(df[feature_attributes])

    df['__outlier__'] = preds
    valid_indices = df[df['__outlier__'] == 1].index

    df_to_return = df.drop(index=df.index.difference(valid_indices))

    # self check
    print(f"Isolation Forest removed {len(df) - len(df)} rows as outliers from '"
          f"{target_attribute}' based on {feature_attributes}")

    return df_to_return


# outlier detection and removal using Standard Deviation (Z-score)
def standard_deviation_detection(df, target_attribute, threshold=3):
    # df_copy = df.dropna(subset=[target_attribute]).copy()
    df_copy = df.copy()
    # df_copy = impute_attribute(df, 'ConvertedCompYearly', 'mean')
    # df_copy = impute_attribute(df, 'EdLevel', 'mode')
    # df_copy = impute_attribute(df, 'ConvertedCompYearly', 'median')
    # df_copy = impute_attribute(df, 'EdLevel', 'mode')

    mean = df_copy[target_attribute].mean()
    std = df_copy[target_attribute].std()

    low = mean - threshold * std
    high = mean + threshold * std

    mask = (df_copy[target_attribute] >= low) & (df_copy[target_attribute] <= high)
    valid_indices = df_copy[mask].index

    df_cleaned = df.drop(index=df_copy.index.difference(valid_indices))

    # self check
    print(f"Standard Deviation method removed {len(df) - len(df_cleaned)} outliers from '{target_attribute}'")

    return df_cleaned

