"""
Data loading, cleaning, caching, and feature engineering module.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Streamlit caching wrapper with graceful fallback for standalone script execution
try:
    import streamlit as st
    cache_decorator = st.cache_data
except ImportError:
    def cache_decorator(func):
        return func


def _resolve_path(filename: str) -> Path:
    """Resolve file path relative to project structure or datasets folder."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    candidates = [
        base_dir / "datasets" / filename,
        Path("datasets") / filename,
        Path(filename),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


@cache_decorator
def load_social_media_data() -> pd.DataFrame:
    """
    Load, clean, and enrich the Social Media Impact on Life dataset.
    Returns:
        pd.DataFrame: Cleaned and feature-engineered dataset.
    """
    path = _resolve_path("Social_media_impact_on_life.csv")
    df = pd.read_csv(path)

    # Impute missing values
    if "Perceived_Stress_Score" in df.columns:
        median_stress = df["Perceived_Stress_Score"].median()
        df["Perceived_Stress_Score"] = df["Perceived_Stress_Score"].fillna(median_stress)

    if "Academic_Performance_GPA" in df.columns:
        median_gpa = df["Academic_Performance_GPA"].median()
        df["Academic_Performance_GPA"] = df["Academic_Performance_GPA"].fillna(median_gpa)

    # Ensure correct data types
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(df["Age"].median()).astype(int)
    df["Daily_Usage_Hours"] = pd.to_numeric(df["Daily_Usage_Hours"], errors="coerce").fillna(0.0)
    df["Weekend_Extra_Hours"] = pd.to_numeric(df["Weekend_Extra_Hours"], errors="coerce").fillna(0.0)
    df["Sleep_Duration_Hours"] = pd.to_numeric(df["Sleep_Duration_Hours"], errors="coerce").fillna(7.0)
    df["Sleep_Quality_Score"] = pd.to_numeric(df["Sleep_Quality_Score"], errors="coerce").fillna(5).astype(int)
    df["Mental_Health_Index"] = pd.to_numeric(df["Mental_Health_Index"], errors="coerce").fillna(5).astype(int)

    # Feature Engineering
    df["Total_Weekly_Hours"] = (df["Daily_Usage_Hours"] * 5) + ((df["Daily_Usage_Hours"] + df["Weekend_Extra_Hours"]) * 2)
    df["Screen_to_Sleep_Ratio"] = (df["Daily_Usage_Hours"] / np.maximum(df["Sleep_Duration_Hours"], 1.0)).round(2)
    df["Composite_Wellbeing_Score"] = ((df["Sleep_Quality_Score"] + df["Mental_Health_Index"]) / 2.0).round(2)

    # Categorical bins
    df["Stress_Category"] = pd.cut(
        df["Perceived_Stress_Score"],
        bins=[-np.inf, 4.0, 7.0, np.inf],
        labels=["Low Stress (1-4)", "Moderate Stress (4-7)", "High Stress (7-10)"]
    )

    df["GPA_Tier"] = pd.cut(
        df["Academic_Performance_GPA"],
        bins=[-np.inf, 2.99, 3.49, np.inf],
        labels=["Below 3.0", "3.0 - 3.5", "Above 3.5"]
    )

    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[14, 17, 21, 30],
        labels=["Adolescents (15-17)", "Young Adults (18-21)", "Adults (22+)"]
    )

    return df


@cache_decorator
def load_sleep_doomscrolling_data() -> pd.DataFrame:
    """
    Load, clean, and enrich the Sleep & Doomscrolling Habits dataset.
    Returns:
        pd.DataFrame: Cleaned and feature-engineered dataset.
    """
    path = _resolve_path("sleep_doomscrolling_habits.csv")
    df = pd.read_csv(path)

    # Impute numeric missing values
    num_imputations = {
        "caffeine_intake_mg_per_day": df["caffeine_intake_mg_per_day"].median(),
        "sleep_quality_score": df["sleep_quality_score"].median(),
        "exercise_minutes_per_day": df["exercise_minutes_per_day"].median(),
        "days_since_last_digital_detox": df["days_since_last_digital_detox"].median(),
        "weekly_sleep_debt_hours": df["weekly_sleep_debt_hours"].median(),
    }
    for col, val in num_imputations.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # Impute categorical missing values
    cat_imputations = {
        "occupation_status": df["occupation_status"].mode()[0] if not df["occupation_status"].empty else "Unknown",
        "primary_device_used_at_night": df["primary_device_used_at_night"].mode()[0] if not df["primary_device_used_at_night"].empty else "Unknown",
        "bedtime_routine_type": df["bedtime_routine_type"].mode()[0] if not df["bedtime_routine_type"].empty else "Screen-based",
    }
    for col, val in cat_imputations.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # Feature Engineering
    df["nightly_doomscroll_time_min"] = df["doomscroll_sessions_per_night"] * df["avg_doomscroll_session_minutes"]
    df["weekly_doomscroll_hours"] = ((df["nightly_doomscroll_time_min"] * 7) / 60.0).round(2)
    df["night_disruption_index"] = df["number_of_night_wakeups"] + df["phone_checks_per_night"]

    # Sleep Debt Tier
    df["sleep_debt_severity"] = pd.cut(
        df["weekly_sleep_debt_hours"],
        bins=[-np.inf, 2.5, 6.0, np.inf],
        labels=["Low Debt (<2.5h)", "Moderate Debt (2.5-6h)", "Severe Debt (>6h)"]
    )

    # Digital detox categories
    df["detox_recency_tier"] = pd.cut(
        df["days_since_last_digital_detox"],
        bins=[-np.inf, 14, 60, np.inf],
        labels=["Recent (≤14 days)", "Moderate (15-60 days)", "Prolonged (>60 days)"]
    )

    # Age group
    df["age_group"] = pd.cut(
        df["age"],
        bins=[14, 20, 30, 40, 60],
        labels=["Teens (15-20)", "Twenties (21-30)", "Thirties (31-40)", "40+"]
    )

    return df


@cache_decorator
def load_social_media_dictionary() -> pd.DataFrame:
    """Load the Social Media Impact data dictionary."""
    path = _resolve_path("Social_media_data_dictionary.csv")
    return pd.read_csv(path)


@cache_decorator
def load_sleep_dictionary() -> pd.DataFrame:
    """Load the Sleep & Doomscrolling data dictionary."""
    path = _resolve_path("sleep_doomscrolling_data_dictionary.csv")
    return pd.read_csv(path)
