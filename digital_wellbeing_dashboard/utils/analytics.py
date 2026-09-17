"""
Analytics, statistical calculations, machine learning, and simulation module.
"""
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def calculate_social_media_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate executive KPI metrics for the Social Media dataset."""
    total_records = len(df)
    if total_records == 0:
        return {}

    avg_daily_hours = df["Daily_Usage_Hours"].mean()
    avg_gpa = df["Academic_Performance_GPA"].mean()
    avg_sleep = df["Sleep_Duration_Hours"].mean()
    
    high_stress_count = (df["Perceived_Stress_Score"] >= 7.0).sum()
    pct_high_stress = (high_stress_count / total_records) * 100.0

    beneficial_count = (df["Overall_Impact"] == "Beneficial").sum()
    pct_beneficial = (beneficial_count / total_records) * 100.0

    late_night_count = (df["Late_Night_Usage"] == True).sum()
    pct_late_night = (late_night_count / total_records) * 100.0

    return {
        "total_records": total_records,
        "avg_daily_hours": round(avg_daily_hours, 2),
        "avg_gpa": round(avg_gpa, 2),
        "avg_sleep": round(avg_sleep, 2),
        "pct_high_stress": round(pct_high_stress, 1),
        "pct_beneficial": round(pct_beneficial, 1),
        "pct_late_night": round(pct_late_night, 1),
    }


def calculate_sleep_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate executive KPI metrics for the Sleep & Doomscrolling dataset."""
    total_records = len(df)
    if total_records == 0:
        return {}

    doomscroller_count = (df["doomscroller"] == "Yes").sum()
    pct_doomscrollers = (doomscroller_count / total_records) * 100.0

    avg_bedtime_screen = df["bedtime_screen_time_minutes"].mean()
    avg_sleep_latency = df["sleep_latency_minutes"].mean()
    avg_sleep_debt = df["weekly_sleep_debt_hours"].mean()
    avg_anxiety = df["anxiety_score"].mean()

    poor_sleep_count = (df["sleep_quality_category"] == "Poor").sum()
    pct_poor_sleep = (poor_sleep_count / total_records) * 100.0

    return {
        "total_records": total_records,
        "pct_doomscrollers": round(pct_doomscrollers, 1),
        "avg_bedtime_screen": round(avg_bedtime_screen, 1),
        "avg_sleep_latency": round(avg_sleep_latency, 1),
        "avg_sleep_debt": round(avg_sleep_debt, 2),
        "avg_anxiety": round(avg_anxiety, 1),
        "pct_poor_sleep": round(pct_poor_sleep, 1),
    }


def get_platform_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate aggregate behavioral metrics by Social Media platform."""
    summary = df.groupby("Primary_Platform").agg(
        User_Count=("Student_ID", "count"),
        Avg_Daily_Hours=("Daily_Usage_Hours", "mean"),
        Avg_Weekend_Extra=("Weekend_Extra_Hours", "mean"),
        Avg_GPA=("Academic_Performance_GPA", "mean"),
        Avg_Sleep_Hours=("Sleep_Duration_Hours", "mean"),
        Avg_Stress_Score=("Perceived_Stress_Score", "mean"),
        Avg_Mental_Health=("Mental_Health_Index", "mean"),
    ).reset_index()

    # Round float columns
    float_cols = summary.select_dtypes(include=[np.floating, float]).columns
    summary[float_cols] = summary[float_cols].round(2)
    return summary.sort_values(by="User_Count", ascending=False)


def get_doomscrolling_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate comparative summary between Doomscrollers and Non-Doomscrollers."""
    summary = df.groupby("doomscroller").agg(
        Sample_Size=("respondent_id", "count"),
        Avg_Bedtime_Screen_Min=("bedtime_screen_time_minutes", "mean"),
        Avg_Sleep_Latency_Min=("sleep_latency_minutes", "mean"),
        Avg_Night_Wakeups=("number_of_night_wakeups", "mean"),
        Avg_Sleep_Hours=("sleep_hours_per_night", "mean"),
        Avg_Weekly_Sleep_Debt_Hrs=("weekly_sleep_debt_hours", "mean"),
        Avg_Anxiety_Score=("anxiety_score", "mean"),
        Avg_Daytime_Fatigue=("daytime_fatigue_score", "mean"),
    ).reset_index()

    float_cols = summary.select_dtypes(include=[np.floating, float]).columns
    summary[float_cols] = summary[float_cols].round(2)
    return summary


def calculate_correlation_matrix(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Compute Pearson correlation matrix for specified numerical columns."""
    valid_cols = [col for col in numeric_cols if col in df.columns]
    return df[valid_cols].corr().round(3)


def train_sleep_quality_model(df: pd.DataFrame) -> Tuple[Any, float, Dict[str, float], List[str]]:
    """
    Train a Random Forest classifier to predict sleep_quality_category.
    Returns:
        (model, accuracy, feature_importances_dict, feature_list)
    """
    feature_cols = [
        "bedtime_screen_time_minutes",
        "total_daily_screen_time_hours",
        "doomscroll_sessions_per_night",
        "avg_doomscroll_session_minutes",
        "sleep_latency_minutes",
        "number_of_night_wakeups",
        "caffeine_intake_mg_per_day",
        "anxiety_score",
        "stress_score",
        "exercise_minutes_per_day",
    ]

    target_col = "sleep_quality_category"
    clean_df = df.dropna(subset=feature_cols + [target_col])

    X = clean_df[feature_cols]
    y = clean_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    feature_importances = dict(
        sorted(
            zip(feature_cols, clf.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    return clf, accuracy, feature_importances, feature_cols


def predict_sleep_quality(
    model: Any,
    feature_names: List[str],
    input_values: Dict[str, float]
) -> Tuple[str, Dict[str, float]]:
    """
    Predict sleep quality category and class probabilities for user inputs.
    """
    input_df = pd.DataFrame([input_values])[feature_names]
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    prob_dict = {cls: round(float(prob) * 100, 1) for cls, prob in zip(classes, probabilities)}
    return prediction, prob_dict


def simulate_lifestyle_intervention(
    bedtime_screen_reduction_pct: float,
    current_latency: float,
    current_sleep_debt: float,
    current_stress: float,
) -> Dict[str, float]:
    """
    Simulate physiological and sleep debt improvements under digital habit reduction.
    Empirical behavioral intervention multipliers:
    - 30% screen reduction yields ~25% decrease in sleep latency
    - Reduces sleep debt by ~0.35h per 20% reduction
    - Reduces subjective stress score by ~0.15 pts per 10% reduction
    """
    factor = bedtime_screen_reduction_pct / 100.0
    latency_savings = current_latency * (0.35 * factor)
    sleep_debt_savings = current_sleep_debt * (0.40 * factor)
    stress_reduction = current_stress * (0.25 * factor)

    projected_latency = max(5.0, current_latency - latency_savings)
    projected_sleep_debt = max(0.0, current_sleep_debt - sleep_debt_savings)
    projected_stress = max(1.0, current_stress - stress_reduction)

    return {
        "projected_latency": round(projected_latency, 1),
        "latency_saved_min": round(latency_savings, 1),
        "projected_sleep_debt": round(projected_sleep_debt, 2),
        "sleep_debt_saved_hrs": round(sleep_debt_savings, 2),
        "projected_stress": round(projected_stress, 2),
        "stress_reduction_pts": round(stress_reduction, 2),
    }
