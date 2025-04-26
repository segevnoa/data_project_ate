import numpy as np
import pandas as pd
import utilities
from handle_missing_values import *
from handle_outliers import *

# load dataset
PATH = '/Users/noasegev/StackOverflow.csv'
df = pd.read_csv(PATH)
df = utilities.bin_country_into_continent(df)

confounders_to_edlevel = [
    'Age',
    'Continent',
    'Ethnicity',
    'Gender'
]

# Missing Values (no outlier detection)
print('ate calculations for missing values:')
# # removal
dropna_df = df.copy()
dropna_df = dropna_df.dropna()
dropna_df['EdLevel'] = dropna_df['EdLevel'].apply(utilities.bin_education_level)
dropna_df['EdLevel'] = dropna_df['EdLevel'].map({'Low': 0, 'High': 1})
print('drop na method:')
dropna_ate = utilities.calculate_ate(dropna_df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel)
print(dropna_ate)

# mean imputation
mean_impute_df = df.copy()
mean_impute_df = impute_attribute(mean_impute_df, 'ConvertedCompYearly', 'mean')
mean_impute_df = impute_attribute(mean_impute_df, 'EdLevel', 'mode')
mean_impute_df['EdLevel'] = mean_impute_df['EdLevel'].apply(utilities.bin_education_level)
mean_impute_df['EdLevel'] = mean_impute_df['EdLevel'].map({'Low': 0, 'High': 1})
print('mean + mode imputation method:')
mean_impute_ate = utilities.calculate_ate(mean_impute_df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel)
print(mean_impute_ate)

# median imputation
median_impute_df = df.copy()
median_impute_df = impute_attribute(median_impute_df, 'ConvertedCompYearly', 'median')
median_impute_df = impute_attribute(median_impute_df, 'EdLevel', 'mode')
median_impute_df['EdLevel'] = median_impute_df['EdLevel'].apply(utilities.bin_education_level)
median_impute_df['EdLevel'] = median_impute_df['EdLevel'].map({'Low': 0, 'High': 1})
print('median + mode imputation method:')
median_impute_ate = utilities.calculate_ate(median_impute_df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel)
print(median_impute_ate)

# fancy impute
fancy_impute_df = df.copy()
fancy_impute_df = utilities.encode_demographics(fancy_impute_df)
fancy_impute_df = utilities.encode_edlevel(fancy_impute_df)
# fancy_impute_df['EdLevel'] = fancy_impute_df['EdLevel'].apply(utilities.bin_education_level)
# fancy_impute_df['EdLevel'] = fancy_impute_df['EdLevel'].map({'Low': 0, 'High': 1})
fancy_impute_df = iterative_impute(fancy_impute_df, 'ConvertedCompYearly', confounders_to_edlevel)
fancy_impute_df = iterative_impute(fancy_impute_df, 'EdLevel', confounders_to_edlevel)
fancy_impute_df['EdLevel'] = fancy_impute_df['EdLevel'].apply(lambda x: 1 if x >= 2 else 0)
fancy_impute_df = fancy_impute_df.dropna() # for the missing values in other attributes
print("fancy impute method:")
fancy_impute_ate = utilities.calculate_ate(fancy_impute_df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel)
print(fancy_impute_ate)
