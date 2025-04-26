from dowhy import CausalModel
import numpy as np
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

    # initiate model
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

    print("Estimated ATE:", estimate.value)

    return estimate.value
