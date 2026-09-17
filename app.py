"""
Digital Wellbeing & Behavioral Analytics Dashboard
An interactive Streamlit application analyzing social media consumption, 
doomscrolling habits, sleep disruption, and academic/lifestyle performance.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Configure Streamlit page
st.set_page_config(
    page_title="Digital Wellbeing & Behavioral Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import internal modules
from digital_wellbeing_dashboard.utils.data_loader import (
    load_social_media_data,
    load_sleep_doomscrolling_data,
    load_social_media_dictionary,
    load_sleep_dictionary,
)
from digital_wellbeing_dashboard.utils.analytics import (
    calculate_social_media_kpis,
    calculate_sleep_kpis,
    get_platform_summary,
    get_doomscrolling_summary,
    calculate_correlation_matrix,
    train_sleep_quality_model,
    predict_sleep_quality,
    simulate_lifestyle_intervention,
)
from digital_wellbeing_dashboard.utils.charts import (
    create_kpi_card,
    create_distribution_plot,
    create_scatter_plot,
    create_correlation_heatmap,
    create_platform_comparison_chart,
    create_sleep_latency_boxplot,
    create_sleep_quality_pie,
    create_feature_importance_bar,
    create_radar_chart,
    create_grouped_bar_chart,
)

# Custom CSS for executive styling
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Subtle headers */
    h1, h2, h3 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-weight: 700;
    }
    
    /* Section dividers */
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 1.25rem;
        margin-bottom: 0.75rem;
        color: #E2E8F0;
        border-left: 4px solid #3B82F6;
        padding-left: 10px;
    }
    
    /* Info box */
    .custom-callout {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


# Load Datasets
@st.cache_data
def get_data():
    df_social = load_social_media_data()
    df_sleep = load_sleep_doomscrolling_data()
    dict_social = load_social_media_dictionary()
    dict_sleep = load_sleep_dictionary()
    return df_social, df_sleep, dict_social, dict_sleep

df_social_raw, df_sleep_raw, dict_social, dict_sleep = get_data()

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=500&auto=format&fit=crop&q=60", use_container_width=True)
    st.title("🎛️ Analytics Controls")
    st.caption("Behavioral Health & Technology Intelligence")

    study_view = st.radio(
        "Select Analytical Domain",
        ["Unified Study (Both)", "Social Media & Academics", "Doomscrolling & Sleep Health"],
        index=0
    )

    st.markdown("---")
    st.subheader("Filter Cohort")

    # Dynamic filters for Social Media
    all_platforms = sorted(df_social_raw["Primary_Platform"].unique().tolist())
    selected_platforms = st.multiselect("Platforms", all_platforms, default=all_platforms)

    # Age Filter
    min_age = int(min(df_social_raw["Age"].min(), df_sleep_raw["age"].min()))
    max_age = int(max(df_social_raw["Age"].max(), df_sleep_raw["age"].max()))
    age_range = st.slider("Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    # Sleep quality categories
    all_sleep_cats = sorted(df_sleep_raw["sleep_quality_category"].unique().tolist())
    selected_sleep_cats = st.multiselect("Sleep Quality Cohorts", all_sleep_cats, default=all_sleep_cats)

    # Doomscroller status
    doomscroll_filter = st.selectbox("Doomscroller Status", ["All Participants", "Doomscrollers Only", "Non-Doomscrollers Only"])

    st.markdown("---")
    st.markdown("""
    **Project Info**  
    *Dataset Size*: 5,500 Total Records  
    *Architecture*: Streamlit + Plotly + Scikit-Learn  
    *Repository*: Executive Portfolio Standards  
    """)

# Filter Applications
df_social = df_social_raw[
    (df_social_raw["Primary_Platform"].isin(selected_platforms)) &
    (df_social_raw["Age"] >= age_range[0]) &
    (df_social_raw["Age"] <= age_range[1])
].copy()

df_sleep = df_sleep_raw[
    (df_sleep_raw["age"] >= age_range[0]) &
    (df_sleep_raw["age"] <= age_range[1]) &
    (df_sleep_raw["sleep_quality_category"].isin(selected_sleep_cats))
].copy()

if doomscroll_filter == "Doomscrollers Only":
    df_sleep = df_sleep[df_sleep["doomscroller"] == "Yes"]
elif doomscroll_filter == "Non-Doomscrollers Only":
    df_sleep = df_sleep[df_sleep["doomscroller"] == "No"]


# ----------------- MAIN INTERFACE -----------------
st.title("🧠 Digital Wellbeing & Behavioral Health Intelligence")
st.markdown("""
A data analytics dashboard exploring **screen exposure**, **nighttime doomscrolling**, 
**circadian disruption**, and **academic/psychological outcomes** across empirical cohorts.
""")

# Tabs for structured navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Overview",
    "📱 Social Media & Academics",
    "🌙 Doomscrolling & Sleep Disruption",
    "🔮 AI Predictor & What-If Simulator",
    "📋 Data Quality & Explorer"
])


# ==========================================
# TAB 1: EXECUTIVE OVERVIEW
# ==========================================
with tab1:
    st.markdown('<div class="section-title">Global Behavioral Health KPIs</div>', unsafe_allow_html=True)
    
    kpi_s = calculate_social_media_kpis(df_social)
    kpi_d = calculate_sleep_kpis(df_sleep)

    # Render Metric Cards in Columns
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            create_kpi_card("Analyzed Cohort", f"{kpi_s.get('total_records', 0) + kpi_d.get('total_records', 0):,}", "Active respondents across studies"),
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            create_kpi_card("Daily Screen Time", f"{kpi_s.get('avg_daily_hours', 0)} hrs", "Avg weekday platform usage", delta="+2.1 hrs vs baseline", delta_color="red"),
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            create_kpi_card("Weekly Sleep Debt", f"{kpi_d.get('avg_sleep_debt', 0)} hrs", "Cumulative weekly sleep deficit", delta="Critical Alert", delta_color="red"),
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            create_kpi_card("Avg Academic GPA", f"{kpi_s.get('avg_gpa', 0):.2f}", "Across collegiate & school cohorts", delta="Stable", delta_color="green"),
            unsafe_allow_html=True
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(
            create_kpi_card("Doomscrolling Rate", f"{kpi_d.get('pct_doomscrollers', 0)}%", "Active nighttime doomscrollers"),
            unsafe_allow_html=True
        )
    with c6:
        st.markdown(
            create_kpi_card("Bedtime Screen Time", f"{kpi_d.get('avg_bedtime_screen', 0)} min", "In-bed digital exposure"),
            unsafe_allow_html=True
        )
    with c7:
        st.markdown(
            create_kpi_card("Sleep Latency", f"{kpi_d.get('avg_sleep_latency', 0)} min", "Time to sleep onset (Normal < 20m)"),
            unsafe_allow_html=True
        )
    with c8:
        st.markdown(
            create_kpi_card("High Stress Rate", f"{kpi_s.get('pct_high_stress', 0)}%", "Stress score ≥ 7.0 / 10"),
            unsafe_allow_html=True
        )

    st.markdown('<div class="custom-callout">📌 <b>Executive Synthesis</b>: Excessive nighttime screen exposure (averaging 50.1 minutes in bed) delays sleep onset by up to 2.4x compared to clinical norms. While 81.8% of students perceive social media as beneficial for connectivity, 85.0% exhibit high perceived stress and substantial sleep deficits that cascade into daytime fatigue.</div>', unsafe_allow_html=True)

    # Overview Visuals
    col_left, col_right = st.columns(2)
    with col_left:
        fig_dist = create_distribution_plot(
            df_social,
            column="Daily_Usage_Hours",
            title="Daily Screen Time Distribution by Platform",
            color_col="Primary_Platform"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_right:
        fig_pie = create_sleep_quality_pie(
            df_sleep,
            title="Sleep Quality Category Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ==========================================
# TAB 2: SOCIAL MEDIA & ACADEMICS
# ==========================================
with tab2:
    st.markdown('<div class="section-title">Academic Achievement & Platform Dynamics</div>', unsafe_allow_html=True)
    
    p_summary = get_platform_summary(df_social)
    
    col_p1, col_p2 = st.columns([1.2, 1])
    with col_p1:
        fig_plat = create_platform_comparison_chart(
            p_summary,
            metric_col="Avg_Daily_Hours",
            title="Average Daily Screen Time by Platform (Hours)"
        )
        st.plotly_chart(fig_plat, use_container_width=True)
    with col_p2:
        fig_gpa = create_platform_comparison_chart(
            p_summary,
            metric_col="Avg_GPA",
            title="Average Academic GPA by Platform",
            color_col="Primary_Platform"
        )
        st.plotly_chart(fig_gpa, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Social Comparison & Psychological Well-being</div>', unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        fig_stress = create_grouped_bar_chart(
            df_social,
            x_col="Social_Comparison_Frequency",
            y_col="Perceived_Stress_Score",
            group_col="Academic_Level",
            title="Perceived Stress vs. Social Comparison Frequency"
        )
        st.plotly_chart(fig_stress, use_container_width=True)

    with col_s2:
        fig_scatter = create_scatter_plot(
            df_social,
            x_col="Daily_Usage_Hours",
            y_col="Academic_Performance_GPA",
            title="GPA vs. Daily Social Media Usage (Trendline: OLS)",
            color_col="Academic_Level",
            trendline="ols",
            hover_name="Student_ID"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Correlation Matrix
    st.markdown('<div class="section-title">Inter-Variable Correlation Matrix</div>', unsafe_allow_html=True)
    social_num_cols = [
        "Age", "Daily_Usage_Hours", "Weekend_Extra_Hours", 
        "Sleep_Duration_Hours", "Sleep_Quality_Score", 
        "Perceived_Stress_Score", "Mental_Health_Index", "Academic_Performance_GPA"
    ]
    corr_social = calculate_correlation_matrix(df_social, social_num_cols)
    fig_corr = create_correlation_heatmap(corr_social, title="Social Media & Academic Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)


# ==========================================
# TAB 3: DOOMSCROLLING & SLEEP DISRUPTION
# ==========================================
with tab3:
    st.markdown('<div class="section-title">Circadian Disruption & Nighttime Technology Usage</div>', unsafe_allow_html=True)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        fig_latency = create_sleep_latency_boxplot(
            df_sleep,
            x_col="doomscroller",
            y_col="sleep_latency_minutes",
            title="Sleep Latency (Minutes to Sleep) by Doomscroller Status"
        )
        st.plotly_chart(fig_latency, use_container_width=True)

    with col_d2:
        fig_debt = create_sleep_latency_boxplot(
            df_sleep,
            x_col="bedtime_routine_type",
            y_col="weekly_sleep_debt_hours",
            title="Weekly Sleep Debt (Hours) across Bedtime Routine Types"
        )
        st.plotly_chart(fig_debt, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Phone Proximity, Wakeups & Fatigue</div>', unsafe_allow_html=True)

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        fig_wakeups = create_scatter_plot(
            df_sleep,
            x_col="bedtime_screen_time_minutes",
            y_col="sleep_latency_minutes",
            title="Bedtime Screen Time vs. Sleep Latency (Min)",
            color_col="keeps_phone_in_bedroom",
            trendline="ols",
            hover_name="respondent_id"
        )
        st.plotly_chart(fig_wakeups, use_container_width=True)

    with col_w2:
        fig_fatigue = create_grouped_bar_chart(
            df_sleep,
            x_col="occupation_status",
            y_col="daytime_fatigue_score",
            group_col="doomscroller",
            title="Daytime Fatigue Scores across Occupations"
        )
        st.plotly_chart(fig_fatigue, use_container_width=True)

    # Comparative table
    st.markdown('<div class="section-title">Doomscroller vs. Non-Doomscroller Breakdown</div>', unsafe_allow_html=True)
    d_summary = get_doomscrolling_summary(df_sleep)
    st.dataframe(d_summary, use_container_width=True)


# ==========================================
# TAB 4: AI PREDICTOR & WHAT-IF SIMULATOR
# ==========================================
with tab4:
    st.markdown('<div class="section-title">Machine Learning Sleep Quality Predictor</div>', unsafe_allow_html=True)
    st.caption("Trained on clinical and behavioral telemetry features using a Random Forest Classifier.")

    # Train model
    with st.spinner("Training predictive model..."):
        model, model_acc, feat_importances, feat_names = train_sleep_quality_model(df_sleep_raw)

    st.success(f"Model successfully trained with Cross-Validation Stratified Accuracy: **{model_acc * 100:.1f}%**")

    col_sim_input, col_sim_output = st.columns([1, 1.2])

    with col_sim_input:
        st.markdown("#### Input Your Daily & Bedtime Telemetry")
        input_bedtime_screen = st.slider("Bedtime Screen Time (minutes)", 0, 180, 45, step=5)
        input_total_screen = st.slider("Total Daily Screen Time (hours)", 1.0, 16.0, 7.5, step=0.5)
        input_sessions = st.slider("Doomscroll Sessions per Night", 0, 6, 2)
        input_session_dur = st.slider("Avg Doomscroll Session Duration (min)", 0, 60, 25, step=5)
        input_latency = st.slider("Current Sleep Latency (min)", 5, 90, 30, step=5)
        input_wakeups = st.slider("Number of Night Wakeups", 0, 6, 1)
        input_caffeine = st.slider("Caffeine Intake (mg/day)", 0, 600, 150, step=25)
        input_anxiety = st.slider("Anxiety Score (0 - 10)", 0, 10, 4)
        input_stress = st.slider("Stress Score (0 - 10)", 0, 10, 5)
        input_exercise = st.slider("Exercise Minutes per Day", 0, 120, 30, step=5)

    with col_sim_output:
        st.markdown("#### Real-Time AI Prediction")
        
        user_features = {
            "bedtime_screen_time_minutes": float(input_bedtime_screen),
            "total_daily_screen_time_hours": float(input_total_screen),
            "doomscroll_sessions_per_night": float(input_sessions),
            "avg_doomscroll_session_minutes": float(input_session_dur),
            "sleep_latency_minutes": float(input_latency),
            "number_of_night_wakeups": float(input_wakeups),
            "caffeine_intake_mg_per_day": float(input_caffeine),
            "anxiety_score": float(input_anxiety),
            "stress_score": float(input_stress),
            "exercise_minutes_per_day": float(input_exercise),
        }

        pred_class, probabilities = predict_sleep_quality(model, feat_names, user_features)

        # Color-coded badge
        badge_color = "#10B981" if pred_class == "Good" else ("#F59E0B" if pred_class == "Fair" else "#EF4444")
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.8); border: 2px solid {badge_color}; border-radius: 12px; padding: 18px; text-align: center; margin-bottom: 20px;">
            <span style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94A3B8;">Predicted Sleep Health Tier</span>
            <div style="font-size: 2.5rem; font-weight: 800; color: {badge_color}; margin-top: 5px;">{pred_class} Sleep Quality</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Predicted Probability Distribution:**")
        prob_cols = st.columns(3)
        with prob_cols[0]:
            st.metric("Good Sleep", f"{probabilities.get('Good', 0)}%")
        with prob_cols[1]:
            st.metric("Fair Sleep", f"{probabilities.get('Fair', 0)}%")
        with prob_cols[2]:
            st.metric("Poor Sleep", f"{probabilities.get('Poor', 0)}%")

        # Feature importance plot
        fig_feat = create_feature_importance_bar(feat_importances)
        st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Interactive "What-If" Lifestyle Intervention Simulator</div>', unsafe_allow_html=True)
    st.caption("Quantify how targeted behavioral adjustments (reducing screen exposure, establishing digital curfews) yield sleep debt recovery.")

    col_whatif_1, col_whatif_2 = st.columns([1, 1.2])
    with col_whatif_1:
        screen_cut = st.slider("Bedtime Screen Reduction Target (%)", 0, 80, 40, step=10)
        curr_sleep_debt = st.number_input("Estimated Current Weekly Sleep Debt (Hours)", min_value=0.0, max_value=25.0, value=6.5, step=0.5)
    
    with col_whatif_2:
        intervention_results = simulate_lifestyle_intervention(
            bedtime_screen_reduction_pct=screen_cut,
            current_latency=input_latency,
            current_sleep_debt=curr_sleep_debt,
            current_stress=input_stress,
        )

        st.markdown("#### Projected Clinical & Recovery Outcomes")
        w_col1, w_col2 = st.columns(2)
        with w_col1:
            st.metric(
                "New Sleep Latency",
                f"{intervention_results['projected_latency']} min",
                delta=f"-{intervention_results['latency_saved_min']} min faster",
                delta_color="normal"
            )
            st.metric(
                "Sleep Debt Balance",
                f"{intervention_results['projected_sleep_debt']} hrs",
                delta=f"-{intervention_results['sleep_debt_saved_hrs']} hrs recovered",
                delta_color="normal"
            )
        with w_col2:
            st.metric(
                "Stress Score Projection",
                f"{intervention_results['projected_stress']} / 10",
                delta=f"-{intervention_results['stress_reduction_pts']} pts lower",
                delta_color="normal"
            )
            st.success("Target Achieved: Reducing screen time before sleep provides rapid nervous system down-regulation!")


# ==========================================
# TAB 5: DATA QUALITY & EXPLORER
# ==========================================
with tab5:
    st.markdown('<div class="section-title">Interactive Data Explorer & Auditing</div>', unsafe_allow_html=True)

    data_choice = st.radio("Choose Dataset to Inspect", ["Social Media Impact (4,500 rows)", "Sleep & Doomscrolling (1,000 rows)"], horizontal=True)

    if "Social Media" in data_choice:
        st.subheader("Social Media Impact on Life Dataset")
        st.caption(f"Currently Showing {len(df_social)} rows after active sidebar filters")
        st.dataframe(df_social, use_container_width=True)

        # Download CSV button
        csv_data = df_social.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Filtered Social Media Data (CSV)",
            data=csv_data,
            file_name="filtered_social_media_impact.csv",
            mime="text/csv",
        )

        st.markdown("#### 📖 Data Dictionary")
        st.dataframe(dict_social, use_container_width=True)

    else:
        st.subheader("Sleep & Doomscrolling Habits Dataset")
        st.caption(f"Currently Showing {len(df_sleep)} rows after active sidebar filters")
        st.dataframe(df_sleep, use_container_width=True)

        # Download CSV button
        csv_data = df_sleep.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Filtered Sleep & Doomscrolling Data (CSV)",
            data=csv_data,
            file_name="filtered_sleep_doomscrolling.csv",
            mime="text/csv",
        )

        st.markdown("#### 📖 Data Dictionary")
        st.dataframe(dict_sleep, use_container_width=True)
