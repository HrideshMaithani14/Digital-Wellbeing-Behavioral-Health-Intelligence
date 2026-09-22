<<<<<<< HEAD
# Digital Wellbeing & Behavioral Health Intelligence

A comprehensive **Data Analytics & Machine Learning project** analyzing social media consumption, bedtime doomscrolling, sleep latency, sleep debt, and academic/lifestyle performance across 5,500 survey respondents.

> **Author:** **Hridesh Maithani** — *Data Scientist*  
> **Project:** Data Analytics & Behavioral Intelligence Portfolio Project  
> *Modeled after enterprise data science standards with interactive Streamlit dashboard, modular codebase, and evidence-backed recommendations.*

---

## 📌 Project Overview

Digital connectivity and algorithmic short-form media platforms (Instagram, TikTok, YouTube, Snapchat, Reddit, X) have fundamentally transformed human lifestyle, sleep physiology, and mental health.

This project delivers an empirical, data-driven investigation to answer crucial behavioral questions:

- How does daily and weekend social media exposure impact academic performance (GPA)?
- What is the relationship between social comparison frequency and clinical stress levels?
- How severely does in-bed doomscrolling prolong sleep onset latency and fragment sleep architecture?
- What are the primary lifestyle determinants governing sleep quality classification?
- Can targeted behavioral interventions (e.g., bedtime digital curfews) effectively reverse accumulated sleep debt?

---

## 🎯 Objectives

1. **Audit & Preprocess Multi-Study Data:** Clean, impute, and feature-engineer 5,500 total records across academic and clinical sleep datasets.
2. **Conduct Exploratory Data Analysis (EDA):** Uncover univariate distributions, bivariate correlations, and multi-segment behavioral cohorts.
3. **Evaluate Circadian Disruption:** Measure the specific impact of nighttime phone checks, blue light exposure, and bedroom device proximity on sleep debt.
4. **Train Machine Learning Models:** Develop a multi-class Random Forest classifier to predict sleep quality categories and extract feature importances.
5. **Develop an Interactive Dashboard:** Build an executive-grade Streamlit application featuring live filtering, KPI cards, correlation heatmaps, and a "What-If" intervention simulator.
6. **Formulate Strategic Recommendations:** Deliver actionable, evidence-based directives for educational institutions, app developers, corporate leaders, and individuals.

---

## 📊 Project Components

### 1. Interactive Streamlit Dashboard (`app.py`)
A production-ready, multi-tab analytics application:

| Section | Focus & Key Insights |
|---|---|
| **📊 Executive Overview** | High-level behavioral KPIs, screen time distributions, and macro health summaries. |
| **📱 Social Media & Academics** | Platform usage benchmarks, GPA scatter plots with OLS regression, and social comparison stress dynamics. |
| **🌙 Doomscrolling & Sleep Disruption** | Bedtime screen time vs sleep latency, nighttime wakeups, sleep debt across occupations and bedtime routines. |
| **🔮 AI Predictor & What-If Simulator** | Random Forest sleep quality classifier with live parameter tuning and quantitative intervention simulations. |
| **📋 Data Quality & Explorer** | Filtered dataset viewer, CSV export tools, statistical summaries, and embedded data dictionaries. |

### 2. Jupyter Analytical Notebook (`Digital_Wellbeing_Behavioral_Analytics.ipynb`)
The end-to-end analytical workflow detailing:
- Dataset quality audits and missing value treatment
- Univariate, bivariate, and multivariate visualizations
- Pearson & Spearman correlation analyses
- Statistical hypothesis testing
- Supervised machine learning pipelines & evaluation metrics
- Comprehensive narrative takeaways

### 3. Modular Analytics Utilities (`digital_wellbeing_dashboard/utils/`)
- `data_loader.py`: Streamlit-cached loading pipelines, robust type-casting, median/mode imputation, and feature engineering.
- `analytics.py`: KPI computations, platform aggregations, cross-tabulations, Random Forest model training, and intervention simulation logic.
- `charts.py`: Reusable, publication-grade Plotly chart generators with consistent executive themes, dark-mode styling, and custom HTML KPI cards.

### 4. Strategic Business & Policy Document (`BUSINESS_RECOMMENDATIONS.md`)
Strategic guidance covering academic schedule adjustments, app-level algorithmic friction, corporate "right-to-disconnect" standards, and the evidence-based 3-2-1 bedtime protocol.

---

## 📂 Dataset Architecture

The project integrates two rich behavioral datasets stored in `datasets/`:

| Dataset File | Records | Features | Target Domain |
|---|---:|---:|---|
| `Social_media_impact_on_life.csv` | 4,500 | 16 | Student demographics, daily/weekend hours, GPA, stress, and mental health index |
| `sleep_doomscrolling_habits.csv` | 1,000 | 30 | Bedtime screen minutes, sleep latency, night wakeups, caffeine, sleep debt, and doomscrolling |

### Accompanying Metadata Dictionaries:
- `datasets/Social_media_data_dictionary.csv`: Full definitions, data types, value ranges, and behavioral significance for each social media feature.
- `datasets/sleep_doomscrolling_data_dictionary.csv`: Clinical and behavioral context for all sleep and doomscrolling variables.

---

## 🔍 Key Empirical Findings

1. **Circadian Delay:** Bedtime screen usage averages **50.1 minutes**, causing an average sleep latency of **27.9 minutes** (near clinical insomnia threshold of 30 minutes).
2. **The Doomscrolling Penalty:** Active doomscrollers suffer **4.86 hours of weekly sleep debt** and 2.1 nighttime awakenings, compared to 1.2 hours of sleep debt in non-doomscrollers.
3. **The Social Comparison Trap:** Students reporting "Often" or "Always" comparing themselves to peers on social media have mean perceived stress scores of **8.1/10**, compared to **3.9/10** for those who rarely compare.
4. **Machine Learning Feature Importance:** The Random Forest classifier identifies **Bedtime Screen Time (16.1%)**, **Sleep Latency (14.1%)**, and **Doomscroll Session Duration (11.1%)** as the top 3 physiological predictors of poor sleep quality.
5. **Bedroom Sanctuary Effect:** Keeping smartphones in the bedroom correlates with a **78% increase in nocturnal wakeups** and a **130% surge in midnight phone checking**.

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Python 3.10+
- **Interactive App Framework:** Streamlit
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Interactive Visualizations:** Plotly Express & Plotly Graph Objects
- **Machine Learning & Modeling:** Scikit-Learn (Random Forest, Classification Metrics, Train-Test Split)
- **Statistical Testing:** SciPy, Statsmodels
- **Static Visuals & Audits:** Matplotlib, Seaborn

---

## 🚀 Quickstart & Installation

### 1. Clone or Open the Repository
```bash
cd "D:\Programing\Programing_folder\Program Files\Hridesh_maithani_data_scince_stuff\Intenship project"
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to explore the live application.

---

## 📁 Repository Structure

```
Intenship project/
│
├── datasets/
│   ├── Social_media_impact_on_life.csv          # 4,500 records
│   ├── sleep_doomscrolling_habits.csv          # 1,000 records
│   ├── Social_media_data_dictionary.csv        # Feature descriptions & ranges
│   └── sleep_doomscrolling_data_dictionary.csv # Clinical & behavioral dictionary
│
├── digital_wellbeing_dashboard/
│   ├── __init__.py                             # Package root
│   └── utils/
│       ├── __init__.py                         # Module re-exports
│       ├── data_loader.py                      # Caching, cleaning & feature engineering
│       ├── analytics.py                        # KPI algorithms, ML models & simulator
│       └── charts.py                           # Plotly visualization suite & styled cards
│
├── app.py                                      # Main Streamlit web application
├── requirements.txt                            # Pinned dependencies
├── README.md                                   # Comprehensive project documentation
├── BUSINESS_RECOMMENDATIONS.md                 # Strategic institutional & lifestyle guide
└── Digital_Wellbeing_Behavioral_Analytics.ipynb# Executable end-to-end data analysis notebook
```

---

## 👤 Author & Acknowledgements
- **Author:** **Hridesh Maithani** — *Data Scientist*
- **Role:** Lead Data Analyst,Data Scientist & Behavioral Intelligence Researcher
#
=======
# Digital Wellbeing & Behavioral Health Intelligence

A comprehensive **Data Analytics & Machine Learning project** analyzing social media consumption, bedtime doomscrolling, sleep latency, sleep debt, and academic/lifestyle performance across 5,500 survey respondents.

> **Author:** **Hridesh Maithani** — *Data Scientist*  
> **Project:** Data Analytics & Behavioral Intelligence Portfolio Project  
> *Modeled after enterprise data science standards with interactive Streamlit dashboard, modular codebase, and evidence-backed recommendations.*

---

## 📌 Project Overview

Digital connectivity and algorithmic short-form media platforms (Instagram, TikTok, YouTube, Snapchat, Reddit, X) have fundamentally transformed human lifestyle, sleep physiology, and mental health.

This project delivers an empirical, data-driven investigation to answer crucial behavioral questions:

- How does daily and weekend social media exposure impact academic performance (GPA)?
- What is the relationship between social comparison frequency and clinical stress levels?
- How severely does in-bed doomscrolling prolong sleep onset latency and fragment sleep architecture?
- What are the primary lifestyle determinants governing sleep quality classification?
- Can targeted behavioral interventions (e.g., bedtime digital curfews) effectively reverse accumulated sleep debt?

---

## 🎯 Objectives

1. **Audit & Preprocess Multi-Study Data:** Clean, impute, and feature-engineer 5,500 total records across academic and clinical sleep datasets.
2. **Conduct Exploratory Data Analysis (EDA):** Uncover univariate distributions, bivariate correlations, and multi-segment behavioral cohorts.
3. **Evaluate Circadian Disruption:** Measure the specific impact of nighttime phone checks, blue light exposure, and bedroom device proximity on sleep debt.
4. **Train Machine Learning Models:** Develop a multi-class Random Forest classifier to predict sleep quality categories and extract feature importances.
5. **Develop an Interactive Dashboard:** Build an executive-grade Streamlit application featuring live filtering, KPI cards, correlation heatmaps, and a "What-If" intervention simulator.
6. **Formulate Strategic Recommendations:** Deliver actionable, evidence-based directives for educational institutions, app developers, corporate leaders, and individuals.

---

## 📊 Project Components

### 1. Interactive Streamlit Dashboard (`app.py`)
A production-ready, multi-tab analytics application:

| Section | Focus & Key Insights |
|---|---|
| **📊 Executive Overview** | High-level behavioral KPIs, screen time distributions, and macro health summaries. |
| **📱 Social Media & Academics** | Platform usage benchmarks, GPA scatter plots with OLS regression, and social comparison stress dynamics. |
| **🌙 Doomscrolling & Sleep Disruption** | Bedtime screen time vs sleep latency, nighttime wakeups, sleep debt across occupations and bedtime routines. |
| **🔮 AI Predictor & What-If Simulator** | Random Forest sleep quality classifier with live parameter tuning and quantitative intervention simulations. |
| **📋 Data Quality & Explorer** | Filtered dataset viewer, CSV export tools, statistical summaries, and embedded data dictionaries. |

### 2. Jupyter Analytical Notebook (`Digital_Wellbeing_Behavioral_Analytics.ipynb`)
The end-to-end analytical workflow detailing:
- Dataset quality audits and missing value treatment
- Univariate, bivariate, and multivariate visualizations
- Pearson & Spearman correlation analyses
- Statistical hypothesis testing
- Supervised machine learning pipelines & evaluation metrics
- Comprehensive narrative takeaways

### 3. Modular Analytics Utilities (`digital_wellbeing_dashboard/utils/`)
- `data_loader.py`: Streamlit-cached loading pipelines, robust type-casting, median/mode imputation, and feature engineering.
- `analytics.py`: KPI computations, platform aggregations, cross-tabulations, Random Forest model training, and intervention simulation logic.
- `charts.py`: Reusable, publication-grade Plotly chart generators with consistent executive themes, dark-mode styling, and custom HTML KPI cards.

### 4. Strategic Business & Policy Document (`BUSINESS_RECOMMENDATIONS.md`)
Strategic guidance covering academic schedule adjustments, app-level algorithmic friction, corporate "right-to-disconnect" standards, and the evidence-based 3-2-1 bedtime protocol.

---

## 📂 Dataset Architecture

The project integrates two rich behavioral datasets stored in `datasets/`:

| Dataset File | Records | Features | Target Domain |
|---|---:|---:|---|
| `Social_media_impact_on_life.csv` | 4,500 | 16 | Student demographics, daily/weekend hours, GPA, stress, and mental health index |
| `sleep_doomscrolling_habits.csv` | 1,000 | 30 | Bedtime screen minutes, sleep latency, night wakeups, caffeine, sleep debt, and doomscrolling |

### Accompanying Metadata Dictionaries:
- `datasets/Social_media_data_dictionary.csv`: Full definitions, data types, value ranges, and behavioral significance for each social media feature.
- `datasets/sleep_doomscrolling_data_dictionary.csv`: Clinical and behavioral context for all sleep and doomscrolling variables.

---

## 🔍 Key Empirical Findings

1. **Circadian Delay:** Bedtime screen usage averages **50.1 minutes**, causing an average sleep latency of **27.9 minutes** (near clinical insomnia threshold of 30 minutes).
2. **The Doomscrolling Penalty:** Active doomscrollers suffer **4.86 hours of weekly sleep debt** and 2.1 nighttime awakenings, compared to 1.2 hours of sleep debt in non-doomscrollers.
3. **The Social Comparison Trap:** Students reporting "Often" or "Always" comparing themselves to peers on social media have mean perceived stress scores of **8.1/10**, compared to **3.9/10** for those who rarely compare.
4. **Machine Learning Feature Importance:** The Random Forest classifier identifies **Bedtime Screen Time (16.1%)**, **Sleep Latency (14.1%)**, and **Doomscroll Session Duration (11.1%)** as the top 3 physiological predictors of poor sleep quality.
5. **Bedroom Sanctuary Effect:** Keeping smartphones in the bedroom correlates with a **78% increase in nocturnal wakeups** and a **130% surge in midnight phone checking**.

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Python 3.10+
- **Interactive App Framework:** Streamlit
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Interactive Visualizations:** Plotly Express & Plotly Graph Objects
- **Machine Learning & Modeling:** Scikit-Learn (Random Forest, Classification Metrics, Train-Test Split)
- **Statistical Testing:** SciPy, Statsmodels
- **Static Visuals & Audits:** Matplotlib, Seaborn

---

## 🚀 Quickstart & Installation

### 1. Clone or Open the Repository
```bash
cd "D:\Programing\Programing_folder\Program Files\Hridesh_maithani_data_scince_stuff\Intenship project"
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to explore the live application.

---

## 📁 Repository Structure

```
Intenship project/
│
├── datasets/
│   ├── Social_media_impact_on_life.csv          # 4,500 records
│   ├── sleep_doomscrolling_habits.csv          # 1,000 records
│   ├── Social_media_data_dictionary.csv        # Feature descriptions & ranges
│   └── sleep_doomscrolling_data_dictionary.csv # Clinical & behavioral dictionary
│
├── digital_wellbeing_dashboard/
│   ├── __init__.py                             # Package root
│   └── utils/
│       ├── __init__.py                         # Module re-exports
│       ├── data_loader.py                      # Caching, cleaning & feature engineering
│       ├── analytics.py                        # KPI algorithms, ML models & simulator
│       └── charts.py                           # Plotly visualization suite & styled cards
│
├── app.py                                      # Main Streamlit web application
├── requirements.txt                            # Pinned dependencies
├── README.md                                   # Comprehensive project documentation
├── BUSINESS_RECOMMENDATIONS.md                 # Strategic institutional & lifestyle guide
└── Digital_Wellbeing_Behavioral_Analytics.ipynb# Executable end-to-end data analysis notebook
```

---

## 👤 Author & Acknowledgements
- **Author:** **Hridesh Maithani** — *Data Scientist*
- **Role:** Lead Data Analyst,Data Scientist & Behavioral Intelligence Researcher
#
>>>>>>> cadb17e695e663d128f34519d07efd7fe7909a44
