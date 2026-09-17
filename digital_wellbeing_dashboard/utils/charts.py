"""
Interactive visualization library powered by Plotly for the Digital Wellbeing Dashboard.
"""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Professional theme configuration
THEME_PALETTE = {
    "primary": "#3B82F6",      # Bright Blue
    "secondary": "#10B981",    # Emerald Green
    "accent": "#F59E0B",       # Warm Amber
    "danger": "#EF4444",       # Rose Red
    "purple": "#8B5CF6",       # Deep Purple
    "teal": "#14B8A6",         # Cyan / Teal
    "background": "rgba(0,0,0,0)",
    "text": "#E2E8F0",
    "grid": "rgba(255, 255, 255, 0.08)",
}

COLOR_SEQUENCE = [
    "#3B82F6", "#10B981", "#F59E0B", "#EF4444", 
    "#8B5CF6", "#06B6D4", "#EC4899", "#84CC16"
]


def _apply_layout_styling(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply unified styling, typography, and responsive margins to Plotly figures."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>" if title else "",
            "font": {"size": 16, "family": "Inter, Segoe UI, sans-serif"},
            "x": 0.02,
            "xanchor": "left",
        },
        template="plotly_dark",
        paper_bgcolor=THEME_PALETTE["background"],
        plot_bgcolor=THEME_PALETTE["background"],
        margin=dict(l=40, r=30, t=50, b=40),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_size=12,
            font_family="Inter, Segoe UI, sans-serif"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)"
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor=THEME_PALETTE["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=THEME_PALETTE["grid"])
    return fig


def create_distribution_plot(
    df: pd.DataFrame,
    column: str,
    title: str,
    color_col: Optional[str] = None,
    nbins: int = 30
) -> go.Figure:
    """Create a styled histogram with marginal box plot."""
    fig = px.histogram(
        df,
        x=column,
        color=color_col,
        nbins=nbins,
        marginal="box",
        color_discrete_sequence=COLOR_SEQUENCE,
        opacity=0.85,
    )
    return _apply_layout_styling(fig, title)


def create_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color_col: Optional[str] = None,
    trendline: Optional[str] = "ols",
    hover_name: Optional[str] = None
) -> go.Figure:
    """Create a bivariate scatter plot with optional regression trendline."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_name=hover_name,
        trendline=trendline,
        color_discrete_sequence=COLOR_SEQUENCE,
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))
    return _apply_layout_styling(fig, title)


def create_correlation_heatmap(corr_df: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    """Create an annotated correlation heatmap."""
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Correlation"))
    return _apply_layout_styling(fig, title)


def create_platform_comparison_chart(
    summary_df: pd.DataFrame,
    metric_col: str,
    title: str,
    color_col: Optional[str] = None
) -> go.Figure:
    """Horizontal or vertical bar chart comparing social platforms."""
    fig = px.bar(
        summary_df,
        x="Primary_Platform",
        y=metric_col,
        color=color_col or "Primary_Platform",
        text=metric_col,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    return _apply_layout_styling(fig, title)


def create_sleep_latency_boxplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color_col: Optional[str] = None
) -> go.Figure:
    """Box and strip distribution plot for sleep latency and disruption."""
    fig = px.box(
        df,
        x=x_col,
        y=y_col,
        color=color_col or x_col,
        points="outliers",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    return _apply_layout_styling(fig, title)


def create_sleep_quality_pie(df: pd.DataFrame, title: str = "Sleep Quality Breakdown") -> go.Figure:
    """Donut chart for categorical breakdown."""
    counts = df["sleep_quality_category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]

    color_map = {
        "Good": "#10B981",
        "Fair": "#F59E0B",
        "Poor": "#EF4444"
    }

    fig = px.pie(
        counts,
        names="Category",
        values="Count",
        hole=0.55,
        color="Category",
        color_discrete_map=color_map,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _apply_layout_styling(fig, title)


def create_feature_importance_bar(
    importances_dict: Dict[str, float],
    title: str = "Top Determinants of Sleep Health (Feature Importance)"
) -> go.Figure:
    """Horizontal bar plot of machine learning model feature importances."""
    features = list(importances_dict.keys())[::-1]
    scores = [importances_dict[k] for k in features]

    # Clean label text
    clean_labels = [f.replace("_", " ").title() for f in features]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=clean_labels,
            orientation="h",
            marker=dict(
                color=scores,
                colorscale="Viridis",
            ),
            text=[f"{s * 100:.1f}%" for s in scores],
            textposition="outside",
        )
    )
    fig.update_layout(xaxis_title="Relative Importance Score")
    return _apply_layout_styling(fig, title)


def create_radar_chart(
    categories: List[str],
    values: List[float],
    title: str = "Behavioral Lifestyle Profile"
) -> go.Figure:
    """Radar / spider chart for multidimensional risk profiling."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(59, 130, 246, 0.25)",
            line=dict(color="#3B82F6", width=2),
            marker=dict(size=6, color="#60A5FA"),
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], gridcolor=THEME_PALETTE["grid"]),
            angularaxis=dict(gridcolor=THEME_PALETTE["grid"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
    )
    return _apply_layout_styling(fig, title)


def create_grouped_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str
) -> go.Figure:
    """Grouped bar chart for cross-segmentation comparisons."""
    agg_df = df.groupby([x_col, group_col])[y_col].mean().reset_index()
    fig = px.bar(
        agg_df,
        x=x_col,
        y=y_col,
        color=group_col,
        barmode="group",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    return _apply_layout_styling(fig, title)


def create_kpi_card(
    title: str,
    value: str,
    subtitle: str = "",
    delta: Optional[str] = None,
    delta_color: str = "normal"
) -> str:
    """Generate modern HTML/CSS markup for executive KPI cards in Streamlit."""
    delta_badge = ""
    if delta:
        color_hex = "#10B981" if delta_color == "green" else ("#EF4444" if delta_color == "red" else "#94A3B8")
        delta_badge = f"""<span style="font-size: 0.82rem; font-weight: 600; color: {color_hex}; margin-left: 8px;">{delta}</span>"""

    html = f"""
    <div style="
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        margin-bottom: 12px;
    ">
        <div style="font-size: 0.85rem; color: #94A3B8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
            {title}
        </div>
        <div style="font-size: 1.85rem; font-weight: 700; color: #F8FAFC; display: flex; align-items: baseline;">
            {value} {delta_badge}
        </div>
        <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">
            {subtitle}
        </div>
    </div>
    """
    return html
