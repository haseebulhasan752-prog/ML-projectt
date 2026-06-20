"""
=============================================================================
EDA & Visualization Module
Saves all plots to static/graphs/
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

GRAPH_DIR = os.path.join(os.path.dirname(__file__), '..', 'static', 'graphs')
os.makedirs(GRAPH_DIR, exist_ok=True)

TARGETS = ['CGPA', 'Stress_Level', 'Depression_Score', 'Anxiety_Score']
PALETTE = 'viridis'
sns.set_theme(style='whitegrid', palette=PALETTE)


def save_fig(name: str, dpi: int = 150):
    path = os.path.join(GRAPH_DIR, f'{name}.png')
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  [GRAPH] Saved: {name}.png")
    return path


def plot_target_distributions(df: pd.DataFrame):
    """Distribution of all 4 target variables."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Target Variable Distributions', fontsize=16, fontweight='bold')
    for ax, col in zip(axes.flat, TARGETS):
        sns.histplot(df[col], kde=True, ax=ax, color='steelblue', bins=30)
        ax.set_title(f'Distribution of {col}', fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    plt.tight_layout()
    save_fig('target_distributions')


def plot_correlation_heatmap(df: pd.DataFrame):
    """Full correlation heatmap."""
    # Use only engineered + original numeric features (cap at 40 for readability)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    key_cols     = [c for c in numeric_cols if c not in TARGETS][:36] + TARGETS
    corr         = df[key_cols].corr()

    fig, ax = plt.subplots(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdYlGn', center=0,
                annot=False, linewidths=0.3, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_fig('correlation_heatmap')


def plot_target_correlations(df: pd.DataFrame):
    """Top-10 feature correlations per target."""
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Top-10 Feature Correlations per Target', fontsize=16, fontweight='bold')

    for ax, target in zip(axes.flat, TARGETS):
        features = [c for c in numeric_cols if c != target]
        corr_vals = df[features].corrwith(df[target]).abs().sort_values(ascending=False).head(10)
        colors = ['#d62728' if v > 0.3 else '#1f77b4' for v in corr_vals]
        corr_vals.plot(kind='barh', ax=ax, color=colors)
        ax.set_title(f'Top Features → {target}', fontsize=12)
        ax.set_xlabel('|Correlation|')
        ax.invert_yaxis()
    plt.tight_layout()
    save_fig('target_correlations')


def plot_boxplots(df: pd.DataFrame):
    """Boxplots of key numeric features."""
    key_features = ['CGPA', 'Stress_Level', 'Depression_Score', 'Anxiety_Score',
                    'Mental_Burden', 'Risk_Score', 'Wellness_Score', 'Lifestyle_Index']
    key_features = [f for f in key_features if f in df.columns]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle('Boxplots – Key Features', fontsize=15, fontweight='bold')
    for ax, col in zip(axes.flat, key_features):
        sns.boxplot(y=df[col], ax=ax, color='coral')
        ax.set_title(col, fontsize=10)
    plt.tight_layout()
    save_fig('boxplots')


def plot_pairplot(df: pd.DataFrame):
    """Pairplot of target variables + engineered scores."""
    cols = ['CGPA', 'Stress_Level', 'Depression_Score', 'Anxiety_Score',
            'Mental_Burden', 'Wellness_Score', 'Risk_Score']
    cols = [c for c in cols if c in df.columns]
    sample = df[cols].sample(min(800, len(df)), random_state=42)
    g = sns.pairplot(sample, diag_kind='kde', plot_kws={'alpha': 0.4},
                     corner=True)
    g.fig.suptitle('Pairplot – Targets & Engineered Features', y=1.02,
                   fontsize=14, fontweight='bold')
    save_fig('pairplot')


def plot_feature_histograms(df: pd.DataFrame):
    """Histograms for main features."""
    feature_cols = ['Age', 'CGPA', 'Financial_Stress', 'Semester_Credit_Load',
                    'Sleep_Quality', 'Physical_Activity', 'Diet_Quality',
                    'Social_Support', 'Mental_Burden', 'Wellness_Score']
    feature_cols = [c for c in feature_cols if c in df.columns]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Feature Histograms', fontsize=15, fontweight='bold')
    for ax, col in zip(axes.flat, feature_cols):
        sns.histplot(df[col], ax=ax, kde=True, color='teal', bins=25)
        ax.set_title(col, fontsize=9)
    plt.tight_layout()
    save_fig('feature_histograms')


def plot_gender_distribution(df: pd.DataFrame):
    """Gender distribution bar chart."""
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df['Gender'].value_counts()
    labels = ['Male' if v == 0 else 'Female' for v in counts.index]
    ax.bar(labels, counts.values, color=['#4C72B0', '#DD8452'])
    ax.set_title('Gender Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count')
    plt.tight_layout()
    save_fig('gender_distribution')


def plot_cgpa_vs_stress(df: pd.DataFrame):
    """CGPA vs Stress scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df['CGPA'], df['Stress_Level'],
                         c=df['Depression_Score'], cmap='Reds', alpha=0.5, s=20)
    plt.colorbar(scatter, ax=ax, label='Depression Score')
    ax.set_xlabel('CGPA')
    ax.set_ylabel('Stress Level')
    ax.set_title('CGPA vs Stress Level\n(color = Depression Score)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig('cgpa_vs_stress')


def generate_all_plots(df: pd.DataFrame):
    """Run all EDA visualizations."""
    print("\n[EDA] Generating visualizations...")
    plot_target_distributions(df)
    plot_correlation_heatmap(df)
    plot_target_correlations(df)
    plot_boxplots(df)
    plot_pairplot(df)
    plot_feature_histograms(df)
    plot_gender_distribution(df)
    plot_cgpa_vs_stress(df)
    print(f"[EDA] ✅ All graphs saved to: {GRAPH_DIR}\n")


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from preprocessing.preprocess import full_preprocess
    df = full_preprocess('../dataset/students_mental_health_survey.csv')
    generate_all_plots(df)
