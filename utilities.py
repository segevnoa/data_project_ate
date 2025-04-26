from dowhy import CausalModel
import numpy as np
from handle_missing_values import *
from handle_outliers import *
# everything here is specific for Stack Overflow


# bin country into continent for demographics
def bin_country_into_continent(df):
    country_to_continent_map = {
        'Canada': 'North America',
        'United States of America': 'North America',
        'Mexico': 'North America',
        'Brazil': 'South America',
        'Argentina': 'South America',
        'Germany': 'Europe',
        'France': 'Europe',
        'United Kingdom of Great Britain and Northern Ireland': 'Europe',
        'Netherlands': 'Europe',
        'India': 'Asia',
        'China': 'Asia',
        'Japan': 'Asia',
        'South Korea': 'Asia',
        'Russian Federation': 'Europe',
        'Australia': 'Oceania',
        'Egypt': 'Africa',
        'South Africa': 'Africa',
        'Nigeria': 'Africa',
        'Kenya': 'Africa',
        'Turkey': 'Asia',
        'Indonesia': 'Asia',
        'Israel': 'Asia',
        'Pakistan': 'Asia',
        'Bangladesh': 'Asia',
        'Philippines': 'Asia',
        'Thailand': 'Asia',
        'Iran, Islamic Republic of...': 'Asia',
        'Ukraine': 'Europe',
        'Spain': 'Europe',
        'Italy': 'Europe',
        'Poland': 'Europe',
        'Portugal': 'Europe',
        'Sweden': 'Europe',
        'Norway': 'Europe',
        'Finland': 'Europe',
        'Ireland': 'Europe',
        'Singapore': 'Asia',
        'Viet Nam': 'Asia',
        'Malaysia': 'Asia',
        'New Zealand': 'Oceania',
        'Iraq': 'Asia',
        'Saudi Arabia': 'Asia',
        'United Arab Emirates': 'Asia',
        'Qatar': 'Asia',
        'Oman': 'Asia',
        'Jordan': 'Asia',
        'Lebanon': 'Asia',
        'Syrian Arab Republic': 'Asia',
        'Afghanistan': 'Asia',
        'Kazakhstan': 'Asia',
        'Uzbekistan': 'Asia',
        'Nepal': 'Asia',
        'Sri Lanka': 'Asia',
        'Myanmar': 'Asia',
        'Cambodia': 'Asia',
        'North Korea': 'Asia',
        'Taiwan': 'Asia',
        'Hong Kong (S.A.R.)': 'Asia'
    }
    df['Continent'] = df['Country'].map(country_to_continent_map)
    return df


def bin_ethnicity(df):
    ethnicity_map = {
        'White': 'White',
        'European': 'White',
        'North American': 'White',
        'Black': 'Black',
        'African': 'Black',
        'North African': 'Black',
        'Asian': 'Asian',
        'East Asian': 'Asian',
        'Southeast Asian': 'Asian',
        'South Asian': 'Asian',
        'Central Asian': 'Asian',
        'Indian': 'Asian',
        'Hispanic or Latino/a': 'Latino',
        'Central American': 'Latino',
        'South American': 'Latino',
        'Caribbean': 'Latino',
        'Middle Eastern': 'Middle Eastern',
        'Indigenous (such as Native American or Indigenous Australian)': 'Indigenous',
        'Biracial': 'Mixed',
        'Multiracial': 'Mixed',
        'Pacific Islander': 'Pacific Islander',
        'Ethnoreligious group': np.nan,
        'Or, in your own words:': np.nan,
        'Prefer not to say': np.nan,
        "I don't know": np.nan,
        np.nan: np.nan
    }
    df['Ethnicity'] = df['Ethnicity'].map(ethnicity_map)
    return df


def bin_education(df):
    edlevel_bins = {
        'Primary/elementary school': 'Primary',
        'Secondary school': 'Secondary',
        'Bachelor’s degree': 'BSc',
        'Associate degree (A.A., A.S., etc.)': 'BSc',
        'Master’s degree': 'MSc',
        'Other doctoral degree (Ph.D., Ed.D., etc.)': 'PhD',
        'Professional degree (JD, MD, etc.)': 'PhD',
        'Some college/university study without earning a degree': 'Other',
        'Something else': 'Other'
    }
    df['EdLevel'] = df['EdLevel'].map(edlevel_bins)
    return df


def encode_edlevel(df):
    df = bin_education(df)
    edlevel_map = {
        'Primary': 0,
        'Secondary': 1,
        'BSc': 2,
        'MSc': 3,
        'PhD': 4,
        'Other': np.nan
    }
    if 'EdLevel' in df.columns:
        df['EdLevel'] = df['EdLevel'].map(edlevel_map)
    return df


def encode_age(df):
    age_map = {
        'Under 18 years old': 0,
        '18-24 years old': 1,
        '25-34 years old': 2,
        '35-44 years old': 3,
        '45-54 years old': 4,
        '55-64 years old': 5,
        '65 years or older': 6,
        'Prefer not to say': np.nan
    }
    if 'Age' in df.columns:
        df['Age'] = df['Age'].map(age_map)
    return df


def encode_continent(df):
    continent_map = {
        'North America': 0,
        'Europe': 1,
        'Asia': 2,
        'Oceania': 3,
        'South America': 4,
        'Africa': 5,
        'Prefer not to say': np.nan
    }
    if 'Continent' in df.columns:
        df['Continent'] = df['Continent'].map(continent_map)
    return df


def encode_gender(df):
    gender_map = {
        'Man': 0,
        'Woman': 1,
        'Non-binary, genderqueer, or gender non-conforming': 2,
        'Or, in your own words:': np.nan,
        'Prefer not to say': np.nan
    }
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map(gender_map)
    return df


def encode_ethnicity(df):
    ethnicity_map = {
        'White': 0,
        'Black': 1,
        'Asian': 2,
        'Latino': 3,
        'Middle Eastern': 4,
        'Indigenous': 5,
        'Mixed': 6,
        'Pacific Islander': 7,
        np.nan: np.nan
    }

    if 'Ethnicity' in df.columns:
        df['Ethnicity'] = df['Ethnicity'].map(ethnicity_map)
    return df


def bin_education_level(ed):
    high_ed = {
        "Bachelor’s degree",
        "Associate degree (A.A., A.S., etc.)",
        "Master’s degree",
        "Other doctoral degree (Ph.D., Ed.D., etc.)",
        "Professional degree (JD, MD, etc.)"
    }

    low_ed = {
        "Primary/elementary school",
        "Secondary school",
        "Some college/university study without earning a degree",
        "Something else"
    }
    if ed in high_ed:
        return 'High'
    elif ed in low_ed:
        return 'Low'
    else:
        return 'Unknown'  # fallback just in case


def encode_demographics(df):
    df = encode_age(df)
    df = encode_gender(df)
    df = encode_continent(df)
    df = encode_ethnicity(df)
    return df


def calculate_ate(df, treatment, outcome, confounders,
                  method='backdoor.propensity_score_matching'):
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=confounders
    )

    identified_estimand = model.identify_effect()

    estimate = model.estimate_effect(
        identified_estimand,
        method_name=method
    )
    print(estimate.value)
    return estimate.value


def bootstrap_ate(df, treatment_col, outcome_col, confounders, n_bootstrap=1000, confidence_level=0.95,
                  model_builder=calculate_ate):
    bootstrap_ates = []

    for _ in range(n_bootstrap):
        # 1. Resample with replacement
        bootstrap_sample = df.sample(frac=1, replace=True)

        # 2. Recalculate ATE
        ate = model_builder(bootstrap_sample, treatment_col, outcome_col, confounders)
        bootstrap_ates.append(ate)

    # 3. Calculate confidence interval
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100

    lower_bound = np.percentile(bootstrap_ates, lower_percentile)
    upper_bound = np.percentile(bootstrap_ates, upper_percentile)

    # 4. Statistical significance
    significance = "Yes" if (lower_bound > 0 or upper_bound < 0) else "No"
    print("lower bound:", lower_bound, "upper bound:", upper_bound, "significance:", significance)

    return (lower_bound, upper_bound), significance, bootstrap_ates


def run_all_together(df, treatment, outcome, cofounders, missing_values_method, outlier_method):
    df_copy = df.copy()

    # handle missing values
    if missing_values_method == "dropna":
        df_copy = df_copy.dropna()
    elif missing_values_method == "mean":
        df_copy = impute_attribute(df_copy, outcome, 'mean')
        df_copy = impute_attribute(df_copy, treatment, 'mode')
    elif missing_values_method == "median":
        df_copy = impute_attribute(df_copy, outcome, 'median')
        df_copy = impute_attribute(df_copy, treatment, 'mode')
    elif missing_values_method == "fancy":
        df_copy = encode_demographics(df_copy)
        df_copy = encode_edlevel(df_copy)
        df_copy = iterative_impute(df_copy, outcome,  cofounders)
        df_copy = iterative_impute(df_copy, treatment, cofounders)
        df_copy[treatment] = df_copy[treatment].apply(lambda x: 1 if x >= 2 else 0)

    # handle outliers
    if outlier_method == "iqr":
        df_copy = iqr_removal_method(df_copy, outcome)
    elif outlier_method == "isolation forest":
        if missing_values_method == "fancy":
            df_copy = isolation_forest_detection(df_copy, outcome, cofounders)
        else:
            df_copy = encode_demographics(df_copy)
            df_copy = isolation_forest_detection(df_copy, outcome, cofounders)
    elif outlier_method == "standard deviation":
        df_copy = standard_deviation_detection(df_copy, outcome)
    elif outlier_method == "none":
        df_copy = df_copy

    # prepare for ate calc
    if missing_values_method != "fancy":
        df_copy[treatment] = df_copy[treatment].apply(bin_education_level)
        df_copy = df_copy[df_copy[treatment].isin(['Low', 'High'])]
        df_copy[treatment] = df_copy[treatment].map({'Low': 0, 'High': 1})
        df_copy = df_copy.dropna(subset=[treatment, outcome] + cofounders)
    elif missing_values_method == "fancy":
        df_copy = df_copy.dropna(subset=[treatment, outcome] + cofounders)

    print("missing values method:", missing_values_method, "outlier detection method:", outlier_method)
    # calculations
    ate = calculate_ate(df_copy, treatment, outcome, cofounders)
    print("ATE:", ate)
    ci = bootstrap_ate(df_copy, treatment, outcome, cofounders)
