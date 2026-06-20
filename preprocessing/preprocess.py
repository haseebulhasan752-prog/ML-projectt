"""
=============================================================================
Student Mental Health & Academic Performance Prediction System
Preprocessing & Feature Engineering Module
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# ORDINAL MAPS
# ─────────────────────────────────────────────────────────────────────────────
ORDINAL_MAPS = {
    'Sleep_Quality':             {'Poor': 0, 'Average': 1, 'Good': 2},
    'Physical_Activity':         {'Low': 0, 'Moderate': 1, 'High': 2},
    'Diet_Quality':              {'Poor': 0, 'Average': 1, 'Good': 2},
    'Social_Support':            {'Low': 0, 'Moderate': 1, 'High': 2},
    'Extracurricular_Involvement': {'Low': 0, 'Moderate': 1, 'High': 2},
    'Counseling_Service_Use':    {'Never': 0, 'Occasionally': 1, 'Frequently': 2},
    'Substance_Use':             {'Never': 0, 'Occasionally': 1, 'Frequently': 2},
}

BINARY_MAPS = {
    'Family_History':   {'No': 0, 'Yes': 1},
    'Chronic_Illness':  {'No': 0, 'Yes': 1},
    'Gender':           {'Male': 0, 'Female': 1},
}

ONE_HOT_COLS = ['Course', 'Relationship_Status', 'Residence_Type']


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV dataset."""
    df = pd.read_csv(path)
    print(f"[DATA] Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and basic cleaning."""
    df = df.copy()

    # Fill missing CGPA with median
    df['CGPA'].fillna(df['CGPA'].median(), inplace=True)

    # Fill missing Substance_Use with 'Never'
    df['Substance_Use'].fillna('Never', inplace=True)

    # Fill any remaining categorical nulls with mode
    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Fill any remaining numeric nulls with median
    for col in df.select_dtypes(include='number').columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    null_counts = df.isnull().sum()
    print(f"[CLEAN] Null values after cleaning:\n{null_counts[null_counts > 0]}")
    print(f"[CLEAN] Total nulls remaining: {df.isnull().sum().sum()}")
    return df


def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ordinal, binary, and one-hot encoding."""
    df = df.copy()

    # Ordinal encoding
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            # Fill any unmapped (NaN) with the median of mapped values
            df[col].fillna(df[col].median(), inplace=True)

    # Binary encoding
    for col, mapping in BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            df[col].fillna(0, inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df, columns=ONE_HOT_COLS, drop_first=True)

    # Ensure all columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(0, inplace=True)

    print(f"[ENCODE] Shape after encoding: {df.shape}")
    return df


def clip_outliers(df: pd.DataFrame,
                  cols: list = ['CGPA', 'Stress_Level',
                                'Depression_Score', 'Anxiety_Score']
                  ) -> pd.DataFrame:
    """Clip outliers using 1st and 99th percentile (IQR-safe)."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            p1  = df[col].quantile(0.01)
            p99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=p1, upper=p99)
    print(f"[OUTLIER] Clipped outliers in: {cols}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all engineered features."""
    df = df.copy()

    # ── Wellness Features ──────────────────────────────────────────────────
    df['Wellness_Score'] = (
        df['Sleep_Quality'] + df['Physical_Activity'] + df['Diet_Quality']
    ) / 3.0

    df['Lifestyle_Index'] = (
        df['Sleep_Quality'] * 0.4 +
        df['Physical_Activity'] * 0.35 +
        df['Diet_Quality'] * 0.25
    )

    df['Wellbeing_vs_Burden'] = (
        df['Wellness_Score'] - (df['Stress_Level'] / 10.0)
    )

    # ── Burden Features ────────────────────────────────────────────────────
    df['Mental_Burden'] = (
        df['Stress_Level'] + df['Depression_Score'] + df['Anxiety_Score']
    ) / 3.0

    df['Support_Deficit'] = (
        (10 - df['Social_Support'] * (10/2)) / 10.0
    )

    df['Academic_Pressure'] = (
        df['Semester_Credit_Load'] * df['Stress_Level']
    ) / (29 * 10)   # normalize to ~[0,1]

    # ── Interaction Features ───────────────────────────────────────────────
    df['Sleep_x_Activity']      = df['Sleep_Quality']   * df['Physical_Activity']
    df['Stress_x_Financial']    = df['Stress_Level']    * df['Financial_Stress']
    df['Depression_x_Sleep']    = df['Depression_Score'] * df['Sleep_Quality']
    df['Anxiety_x_Support']     = df['Anxiety_Score']   * df['Social_Support']
    df['Chronic_Financial']     = df['Chronic_Illness'] * df['Financial_Stress']
    df['Age_Credit_interaction'] = df['Age']             * df['Semester_Credit_Load']
    df['CGPA_stress_interaction']= df['CGPA']            * df['Stress_Level']

    # ── Risk Features ──────────────────────────────────────────────────────
    df['Risk_Score'] = (
        df['Depression_Score'] * 0.3 +
        df['Anxiety_Score']    * 0.3 +
        df['Stress_Level']     * 0.2 +
        df['Financial_Stress'] * 0.1 +
        df['Chronic_Illness']  * 0.1
    )

    df['Family_History_Risk'] = df['Family_History'] * df['Risk_Score']

    df['Substance_Counseling_ratio'] = df['Substance_Use'] / (
        df['Counseling_Service_Use'] + 1
    )

    # ── Aggregate Mental Features ──────────────────────────────────────────
    df['Total_Mental_Load'] = (
        df['Stress_Level'] + df['Depression_Score'] + df['Anxiety_Score']
    )

    df['Avg_Mental_Score'] = df['Total_Mental_Load'] / 3.0

    # ── Polynomial Features ────────────────────────────────────────────────
    for col in ['Sleep_Quality', 'Physical_Activity', 'Diet_Quality', 'Social_Support']:
        df[f'{col}_sq'] = df[col] ** 2

    # ── CGPA Variance Features ─────────────────────────────────────────────
    df['CGPA_deviation']   = df['CGPA'] - df['CGPA'].mean()
    df['CGPA_percentile']  = df['CGPA'].rank() / len(df) * 100

    print(f"[FEATURE ENG] Shape after feature engineering: {df.shape}")
    print(f"[FEATURE ENG] New feature count: {df.shape[1]}")
    return df


def full_preprocess(path: str, save_path: str = None) -> pd.DataFrame:
    """Run complete preprocessing pipeline end-to-end."""
    df = load_data(path)
    df = clean_data(df)
    df = encode_data(df)
    df = clip_outliers(df)
    df = engineer_features(df)

    # Final check
    assert df.isnull().sum().sum() == 0, "Nulls still present after preprocessing!"
    assert all(df.dtypes != 'object'), "Non-numeric columns still present!"

    print(f"\n[PIPELINE] ✅ Final dataset shape : {df.shape}")
    print(f"[PIPELINE] ✅ All columns numeric : True")
    print(f"[PIPELINE] ✅ Total null values   : {df.isnull().sum().sum()}")

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"[PIPELINE] ✅ Saved processed CSV  : {save_path}")

    return df


if __name__ == '__main__':
    df = full_preprocess(
        path='../dataset/students_mental_health_survey.csv',
        save_path='../dataset/processed_student_data.csv'
    )
    print(df.head(3))
