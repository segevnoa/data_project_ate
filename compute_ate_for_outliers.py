import numpy as np
import pandas as pd
import utilities
from handle_missing_values import *
from handle_outliers import *

# load dataset
PATH = '/Users/noasegev/StackOverflow.csv'
df = pd.read_csv(PATH)

confounders_to_edlevel = [
    'Age',
    'Continent',
    'Ethnicity',
    'Gender'
]

df = utilities.bin_country_into_continent(df)


# print("iqr detection method")
# iqr_df = df.copy()
# handle missing values (changes)
# removal
# print("remove missing values")
# num_rows_with_na = df.isna().any(axis=1).sum()
# print(num_rows_with_na)
# idr_df = iqr_df.dropna()
# # iqr
# iqr_df = iqr_removal_method(iqr_df, 'ConvertedCompYearly')
# iqr_df['EdLevel'] = iqr_df['EdLevel'].apply(utilities.bin_education_level)
# iqr_df['EdLevel'] = iqr_df['EdLevel'].map({'Low': 0, 'High': 1})
# iqr_df = iqr_df.dropna(subset=['EdLevel', 'ConvertedCompYearly'] + confounders_to_edlevel)
# ate_by_iqr = utilities.calculate_ate(iqr_df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel)
# print(ate_by_iqr)
# mean + mode
# print("mean + mode imputation")
# iqr_df = impute_attribute(iqr_df, 'ConvertedCompYearly', 'mean')
# iqr_df = impute_attribute(iqr_df, 'EdLevel', 'mode')
# # iqr
# iqr_df = iqr_removal_method(iqr_df, 'ConvertedCompYearly')
# iqr_df['EdLevel'] = iqr_df['EdLevel'].apply(utilities.bin_education_level)
# # safety step for high-low only
# iqr_df = iqr_df[iqr_df['EdLevel'].isin(['Low', 'High'])]
# iqr_df['EdLevel'] = iqr_df['EdLevel'].map({'Low': 0, 'High': 1})
# # remove remaining na
# iqr_df = iqr_df.dropna(subset=['EdLevel', 'ConvertedCompYearly'] + confounders_to_edlevel)
# ate_by_iqr = utilities.calculate_ate(iqr_df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel)
# print(ate_by_iqr)
# # median + mode
# print("median + mode imputation")
# iqr_df = impute_attribute(iqr_df, 'ConvertedCompYearly', 'median')
# iqr_df = impute_attribute(iqr_df, 'EdLevel', 'mode')
# # iqr
# iqr_df = iqr_removal_method(iqr_df, 'ConvertedCompYearly')
# iqr_df['EdLevel'] = iqr_df['EdLevel'].apply(utilities.bin_education_level)
# # safety step for high-low only
# # iqr_df = iqr_df[iqr_df['EdLevel'].isin(['Low', 'High'])]
# iqr_df['EdLevel'] = iqr_df['EdLevel'].map({'Low': 0, 'High': 1})
# # remove remaining na
# iqr_df = iqr_df.dropna(subset=['EdLevel', 'ConvertedCompYearly'] + confounders_to_edlevel)
# ate_by_iqr = utilities.calculate_ate(iqr_df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel)
# print(ate_by_iqr)
# fancy impute
# iqr_df = utilities.encode_demographics(iqr_df)
# iqr_df = utilities.encode_edlevel(iqr_df)
# iqr_df = iterative_impute(iqr_df, 'ConvertedCompYearly', confounders_to_edlevel)
# iqr_df = iterative_impute(iqr_df, 'EdLevel', confounders_to_edlevel)
# iqr_df['EdLevel'] = iqr_df['EdLevel'].apply(lambda x: 1 if x >= 2 else 0)
# iqr
# iqr_df = iqr_removal_method(iqr_df, 'ConvertedCompYearly')
# iqr_df['EdLevel'] = iqr_df['EdLevel'].apply(utilities.bin_education_level)
# iqr_df = iqr_df[iqr_df['EdLevel'].isin(['Low', 'High'])]
# iqr_df['EdLevel'] = iqr_df['EdLevel'].map({'Low': 0, 'High': 1})
# iqr_df = iqr_df.dropna(subset=['EdLevel', 'ConvertedCompYearly'] + confounders_to_edlevel)
# print("method 1: iqr:")
# ate_by_iqr = utilities.calculate_ate(iqr_df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel)
# print(ate_by_iqr)

# print("isolation forest detection method")
iso_forest_df = df.copy()
# handle missing values (before the isolation forest)
# # removal
# print("remove missing values")
# iso_forest_df = iso_forest_df.dropna()
# # mean + mode
# print("mean + mode imputation")
# iso_forest_df = impute_attribute(iso_forest_df, 'ConvertedCompYearly', 'mean')
# iso_forest_df = impute_attribute(iso_forest_df, 'EdLevel', 'mode')
# # median + mode
# print("median + mode imputation")
# iso_forest_df = impute_attribute(iso_forest_df, 'ConvertedCompYearly', 'median')
# iso_forest_df = impute_attribute(iso_forest_df, 'EdLevel', 'mode')
# fancy impute
iso_forest_df = utilities.encode_demographics(iso_forest_df)
iso_forest_df = utilities.encode_edlevel(iso_forest_df)
# iso_forest_df['EdLevel'] = iso_forest_df['EdLevel'].apply(utilities.bin_education_level)
# iso_forest_df['EdLevel'] = iso_forest_df['EdLevel'].map({'Low': 0, 'High': 1})
iso_forest_df = iterative_impute(iso_forest_df, 'ConvertedCompYearly', confounders_to_edlevel)
iso_forest_df = iterative_impute(iso_forest_df, 'EdLevel', confounders_to_edlevel)
iso_forest_df['EdLevel'] = iso_forest_df['EdLevel'].apply(lambda x: 1 if x >= 2 else 0)
# isolation forest
# iso_forest_df = utilities.encode_demographics(iso_forest_df)
iso_forest_df = isolation_forest_detection(iso_forest_df, 'ConvertedCompYearly', confounders_to_edlevel)
# iso_forest_df['EdLevel'] = iso_forest_df['EdLevel'].apply(utilities.bin_education_level)
# iso_forest_df = iso_forest_df[iso_forest_df['EdLevel'].isin(['Low', 'High'])]
# iso_forest_df['EdLevel'] = iso_forest_df['EdLevel'].map({'Low': 0, 'High': 1})
iso_forest_df = iso_forest_df.dropna(subset=['EdLevel', 'ConvertedCompYearly'] + confounders_to_edlevel)
iso_forest_ate = utilities.calculate_ate(iso_forest_df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel)
# print("ate result: ", iso_forest_ate)

# print("standard deviation  method")
# std_dev_df = df.copy()
# handle missing values (changes)
# # removal
# std_dev_df = std_dev_df.dropna()
# # mean + mode impute
# std_dev_df = impute_attribute(std_dev_df, 'ConvertedCompYearly', 'mean')
# std_dev_df = impute_attribute(std_dev_df, 'EdLevel', 'mode')
# median + mode impute
# std_dev_df = impute_attribute(std_dev_df, 'ConvertedCompYearly', 'median')
# std_dev_df = impute_attribute(std_dev_df, 'EdLevel', 'mode')
# # fancy impute
# std_dev_df = utilities.encode_demographics(std_dev_df)
# std_dev_df = utilities.encode_edlevel(std_dev_df)
# std_dev_df = iterative_impute(std_dev_df, 'ConvertedCompYearly', confounders_to_edlevel)
# std_dev_df = iterative_impute(std_dev_df, 'EdLevel', confounders_to_edlevel)
# std_dev_df['EdLevel'] = std_dev_df['EdLevel'].apply(lambda x: 1 if x >= 2 else 0)
# # standard deviation detection
# std_dev_df = standard_deviation_detection(std_dev_df, 'ConvertedCompYearly')
# std_dev_df = std_dev_df.dropna(subset=['EdLevel', 'ConvertedCompYearly'] + confounders_to_edlevel)
# # std_dev_df['EdLevel'] = std_dev_df['EdLevel'].apply(utilities.bin_education_level)
# # std_dev_df['EdLevel'] = std_dev_df['EdLevel'].map({'Low': 0, 'High': 1})
# # ate calculation
# std_dev_ate = utilities.calculate_ate(std_dev_df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel)
# print("ate result: ", std_dev_ate)
