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

confounders_to_edlevel = [
    'Age',
    'Continent',
    'Ethnicity',
    'Gender'
]

df = utilities.bin_country_into_continent(df)

# iqr_df = df.copy()
# iqr_df = iqr_removal_method(iqr_df, 'ConvertedCompYearly')
# iqr_df['EdLevel'] = iqr_df['EdLevel'].apply(utilities.bin_education_level)
# iqr_df['EdLevel'] = iqr_df['EdLevel'].map({'Low': 0, 'High': 1})
# # missing values
# iqr_df = utilities.encode_demographics(iqr_df)
# iqr_df = iterative_impute(iqr_df, 'ConvertedCompYearly', confounders_to_edlevel)
# iqr_df = iterative_impute(iqr_df, 'EdLevel', confounders_to_edlevel)
# iqr_df = iqr_df.dropna()
#
#
# print("method 1: iqr:")
# ate_by_iqr = utilities.calculate_ate(iqr_df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel)

iso_forest_df = df.copy()
# iso_forest_df = utilities.encode_demographics(iso_forest_df)
iso_forest_df = isolation_forest_detection(iso_forest_df, 'ConvertedCompYearly', confounders_to_edlevel)
iso_forest_df['EdLevel'] = iso_forest_df['EdLevel'].apply(utilities.bin_education_level)
iso_forest_df['EdLevel'] = iso_forest_df['EdLevel'].map({'Low': 0, 'High': 1})
print("method 2: isolation forest:")
iso_forest_ate = utilities.calculate_ate(iso_forest_df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel)

# std_dev_df = standard_deviation_detection(df, 'ConvertedCompYearly')
# std_dev_df['EdLevel'] = std_dev_df['EdLevel'].apply(utilities.bin_education_level)
# std_dev_df['EdLevel'] = std_dev_df['EdLevel'].map({'Low': 0, 'High': 1})
# print("method 3: standaed deviation:")
# std_dev_df = utilities.calculate_ate(std_dev_df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel)
