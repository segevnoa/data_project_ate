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
