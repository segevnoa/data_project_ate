import numpy as np
import pandas as pd
import utilities
from handle_missing_values import *
from handle_outliers import *

# load dataset
PATH = '/Users/noasegev/StackOverflow.csv'
df = pd.read_csv(PATH)

# causal DAG
DAG = [
    # vertexes:
    'Age;',
    'CodingActivities;',
    'Continent;',
    'ConvertedCompYearly;',
    'Country;',
    'DevType;',
    'EdLevel;',
    'Employment;',
    'Ethnicity;',
    'Gender;',
    'RemoteWork;',
    'WorkExp;',
    'YearsCodePro;',
    # edges:
    'Age -> ConvertedCompYearly;',
    'Age -> CodingActivities;',
    'Age -> DevType;',
    'Age -> YearsCodePro;',
    'Age -> EdLevel;',
    'CodingActivities -> ConvertedCompYearly;',
    'Continent -> ConvertedCompYearly;',
    'Continent -> EdLevel;',
    'Continent -> Ethnicity;',
    'DevType -> ConvertedCompYearly;',
    'DevType -> Employment;',
    'EdLevel -> DevType;',
    'EdLevel -> YearsCodePro;',
    'EdLevel -> ConvertedCompYearly;',
    'EdLevel -> Employment;',
    'Employment -> RemoteWork;',
    'Employment -> ConvertedCompYearly;',
    'Ethnicity -> ConvertedCompYearly;',
    'Ethnicity -> EdLevel;',
    'Ethnicity -> CodingActivities;',
    'Ethnicity -> DevType;',
    'Ethnicity -> YearsCodePro;',
    'Gender -> EdLevel;',
    'Gender -> CodingActivities;',
    'Gender -> DevType;',
    'Gender -> YearsCodePro;',
    'Gender -> ConvertedCompYearly;',
    'WorkExp -> Employment;',
    'WorkExp -> ConvertedCompYearly;',
    'YearsCodePro -> DevType;',
    'YearsCodePro -> ConvertedCompYearly;',
]

causal_graph = """
                    digraph {
                """
for line in DAG:
    causal_graph = causal_graph + line + "\n"
causal_graph = causal_graph + "}"

# prepare the data with the thingy
df = utilities.bin_country_into_continent(df)

confounders_to_edlevel = [
    'Age',
    'Continent',
    'Ethnicity',
    'Gender'
]

# Missing Values:
print('ate calculations for missing values:')
# removal
dropna_df = df.copy()
dropna_df = dropna_df.dropna()
dropna_df['EdLevel'] = dropna_df['EdLevel'].apply(utilities.bin_education_level)
dropna_df['EdLevel'] = dropna_df['EdLevel'].map({'Low': 0, 'High': 1})
print('drop na method:')
dropna_ate = utilities.calculate_ate(dropna_df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel)

# mean imputation
mean_impute_df = impute_attribute(df, 'ConvertedCompYearly', 'mean')
mean_impute_df = impute_attribute(df, 'EdLevel', 'mode')
mean_impute_df['EdLevel'] = mean_impute_df['EdLevel'].apply(utilities.bin_education_level)
mean_impute_df['EdLevel'] = mean_impute_df['EdLevel'].map({'Low': 0, 'High': 1})
print('mean + mode imputation method:')
mean_impute_ate = utilities.calculate_ate(mean_impute_df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel)

# median imputation
median_impute_df = impute_attribute(df, 'ConvertedCompYearly', 'median')
median_impute_df = impute_attribute(df, 'EdLevel', 'mode')
median_impute_df['EdLevel'] = median_impute_df['EdLevel'].apply(utilities.bin_education_level)
median_impute_df['EdLevel'] = median_impute_df['EdLevel'].map({'Low': 0, 'High': 1})
print('median + mode imputation method:')
median_impute_ate = utilities.calculate_ate(median_impute_df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel)
