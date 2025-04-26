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

# # all calculations for drop na :)
# utilities.run_all_together(df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel, "dropna", "none")
# utilities.run_all_together(df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel, "dropna", "iqr")
# utilities.run_all_together(df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel, "dropna", "isolation forest")
# utilities.run_all_together(df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel, "dropna", "standard deviation")
#
# # all calculations for mean + mode :)
# utilities.run_all_together(df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel, "mean", "none")
# utilities.run_all_together(df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel, "mean", "iqr")
# utilities.run_all_together(df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel, "mean", "isolation forest")
# utilities.run_all_together(df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel, "mean", "standard deviation")
#
# # all calculations for median + mode :)
# utilities.run_all_together(df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel, "median", "none")
# utilities.run_all_together(df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel, "median", "iqr")
# utilities.run_all_together(df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel, "median", "isolation forest")
# utilities.run_all_together(df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel, "median", "standard deviation")
#
# # all calculations for fancy :)
# utilities.run_all_together(df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel, "fancy", "none")
# utilities.run_all_together(df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel, "fancy", "iqr")
# utilities.run_all_together(df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel, "fancy", "isolation forest")
# utilities.run_all_together(df, 'EdLevel', 'ConvertedCompYearly', confounders_to_edlevel, "fancy", "standard deviation")

print("all done <3")
