"""
=============================================================================
Student Mental Health & Academic Performance Prediction System
ML Training Script — Linear Regression + KNN (both Regression & Classification)
=============================================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import StandardScaler
from sklearn.model_selection   import train_test_split, cross_val_score, KFold
from sklearn.linear_model      import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors         import KNeighborsRegressor
from sklearn.metrics           import (mean_absolute_error, mean_squared_error,
                                       r2_score)

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from preprocessing.preprocess import full_preprocess
from utils.eda                import generate_all_plots

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(__file__)
DATASET_PATH    = os.path.join(BASE_DIR, 'dataset', 'students_mental_health_survey.csv')
PROCESSED_PATH  = os.path.join(BASE_DIR, 'dataset', 'processed_student_data.csv')
MODEL_DIR       = os.path.join(BASE_DIR, 'trained_models')
GRAPH_DIR       = os.path.join(BASE_DIR, 'static', 'graphs')
METRICS_PATH    = os.path.join(BASE_DIR, 'static', 'model_metrics.json')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

TARGETS = ['CGPA', 'Stress_Level', 'Depression_Score', 'Anxiety_Score']
SEED    = 42
CV_FOLDS = 5


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SELECTION — pick top-K features per target by correlation
# ─────────────────────────────────────────────────────────────────────────────
def select_features(df: pd.DataFrame, target: str, top_k: int = 30) -> list:
    """Return the top-K features most correlated with the target."""
    all_feats = [c for c in df.columns if c not in TARGETS]
    corr = df[all_feats].corrwith(df[target]).abs().sort_values(ascending=False)
    selected = corr.head(top_k).index.tolist()
    print(f"  [FEATURES] {target}: Top-{top_k} selected → {selected[:5]} ...")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(name: str, y_true, y_pred, y_train_true=None, y_train_pred=None) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)

    result = {'model': name, 'MAE': round(mae,4), 'MSE': round(mse,4),
              'RMSE': round(rmse,4), 'R2': round(r2,4)}

    if y_train_true is not None:
        tr_r2 = r2_score(y_train_true, y_train_pred)
        result['Train_R2'] = round(tr_r2, 4)

    print(f"    {name:30s} | MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}", end='')
    if 'Train_R2' in result:
        print(f"  Train-R²={result['Train_R2']:.4f}")
    else:
        print()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FIND BEST KNN K
# ─────────────────────────────────────────────────────────────────────────────
def tune_knn(X_train, y_train, k_range=range(3, 21)) -> int:
    kf   = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    best_k, best_score = 3, -np.inf
    scores = {}
    for k in k_range:
        pipe = Pipeline([('scaler', StandardScaler()),
                         ('knn', KNeighborsRegressor(n_neighbors=k,
                                                      weights='distance'))])
        cv = cross_val_score(pipe, X_train, y_train,
                             cv=kf, scoring='r2').mean()
        scores[k] = cv
        if cv > best_score:
            best_score = cv
            best_k     = k
    print(f"    [KNN TUNE] Best k={best_k}  CV-R²={best_score:.4f}")
    return best_k


# ─────────────────────────────────────────────────────────────────────────────
# PLOT MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def plot_model_comparison(target: str, results: list):
    names  = [r['model'] for r in results]
    r2s    = [r['R2']    for r in results]
    maes   = [r['MAE']   for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Model Comparison — {target}', fontsize=14, fontweight='bold')

    colors = ['#2196F3' if v < max(r2s) else '#4CAF50' for v in r2s]
    axes[0].barh(names, r2s, color=colors)
    axes[0].set_xlabel('R² Score')
    axes[0].set_title('R² Score (higher = better)')
    axes[0].axvline(0, color='black', linewidth=0.8)
    for i, v in enumerate(r2s):
        axes[0].text(max(0, v)+0.01, i, f'{v:.3f}', va='center', fontsize=9)

    axes[1].barh(names, maes, color=['#FF5722' if v > min(maes) else '#8BC34A' for v in maes])
    axes[1].set_xlabel('MAE')
    axes[1].set_title('Mean Absolute Error (lower = better)')
    for i, v in enumerate(maes):
        axes[1].text(v+0.001, i, f'{v:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, f'model_comparison_{target}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [GRAPH] Saved model_comparison_{target}.png")


def plot_actual_vs_predicted(target: str, y_test, y_pred_lr, y_pred_knn):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Actual vs Predicted — {target}', fontsize=14, fontweight='bold')

    for ax, (preds, name) in zip(axes, [(y_pred_lr, 'Linear Regression'),
                                         (y_pred_knn, 'KNN Regressor')]):
        ax.scatter(y_test, preds, alpha=0.4, s=15, color='steelblue')
        lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect Prediction')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(name)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, f'actual_vs_predicted_{target}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [GRAPH] Saved actual_vs_predicted_{target}.png")


def plot_feature_importance(target: str, features: list, model_pipe):
    lr    = model_pipe.named_steps['model']
    scaler = model_pipe.named_steps['scaler']
    coefs = lr.coef_
    feat_imp = pd.Series(np.abs(coefs), index=features).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp.plot(kind='barh', ax=ax, color='teal')
    ax.set_title(f'Feature Importance (|Coefficients|) — {target}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('|Coefficient|')
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, f'feature_importance_{target}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [GRAPH] Saved feature_importance_{target}.png")


def plot_residuals(target: str, y_test, y_pred_lr, y_pred_knn):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Residual Plots — {target}', fontsize=14, fontweight='bold')

    for ax, (preds, name) in zip(axes, [(y_pred_lr, 'Linear Regression'),
                                         (y_pred_knn, 'KNN Regressor')]):
        residuals = y_test - preds
        ax.scatter(preds, residuals, alpha=0.4, s=15, color='purple')
        ax.axhline(0, color='red', linewidth=1.5, linestyle='--')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residual')
        ax.set_title(name)

    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, f'residuals_{target}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [GRAPH] Saved residuals_{target}.png")


def plot_overall_summary(all_metrics: dict):
    """Bar chart comparing best model R² across all 4 targets."""
    targets = list(all_metrics.keys())
    best_r2 = []
    for t in targets:
        r2s = [m['R2'] for m in all_metrics[t]]
        best_r2.append(max(r2s))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(targets, best_r2,
                  color=['#2196F3', '#4CAF50', '#FF9800', '#E91E63'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Best R² Score')
    ax.set_title('Best R² Score per Prediction Target', fontsize=14, fontweight='bold')
    ax.axhline(0.8, color='red', linestyle='--', linewidth=1.2, label='80% threshold')
    ax.legend()
    for bar, val in zip(bars, best_r2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, 'overall_r2_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print("[GRAPH] Saved overall_r2_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train_all(df: pd.DataFrame) -> dict:
    all_metrics  = {}
    best_models  = {}   # key → pipeline saved to disk
    feature_sets = {}

    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"  TARGET: {target}")
        print(f"{'='*60}")

        # Feature selection
        features = select_features(df, target, top_k=30)
        feature_sets[target] = features
        X = df[features].values
        y = df[target].values

        # Train / test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=SEED)

        results = []

        # ── 1. Linear Regression ──────────────────────────────────────────
        print("\n  [A] Linear Regression")
        lr_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model',  LinearRegression())
        ])
        lr_pipe.fit(X_train, y_train)
        y_pred_lr_train = lr_pipe.predict(X_train)
        y_pred_lr       = lr_pipe.predict(X_test)
        cv_lr = cross_val_score(lr_pipe, X_train, y_train,
                                cv=CV_FOLDS, scoring='r2').mean()
        r = evaluate('Linear Regression', y_test, y_pred_lr,
                     y_train, y_pred_lr_train)
        r['CV_R2'] = round(cv_lr, 4)
        results.append(r)

        # ── 2. Ridge Regression ───────────────────────────────────────────
        print("  [B] Ridge Regression")
        ridge_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model',  Ridge(alpha=1.0))
        ])
        ridge_pipe.fit(X_train, y_train)
        y_pred_ridge_train = ridge_pipe.predict(X_train)
        y_pred_ridge       = ridge_pipe.predict(X_test)
        r2 = evaluate('Ridge Regression', y_test, y_pred_ridge,
                      y_train, y_pred_ridge_train)
        results.append(r2)

        # ── 3. Lasso Regression ───────────────────────────────────────────
        print("  [C] Lasso Regression")
        lasso_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model',  Lasso(alpha=0.01, max_iter=5000))
        ])
        lasso_pipe.fit(X_train, y_train)
        y_pred_lasso = lasso_pipe.predict(X_test)
        r3 = evaluate('Lasso Regression', y_test, y_pred_lasso)
        results.append(r3)

        # ── 4. ElasticNet ─────────────────────────────────────────────────
        print("  [D] ElasticNet")
        en_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model',  ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000))
        ])
        en_pipe.fit(X_train, y_train)
        y_pred_en = en_pipe.predict(X_test)
        r4 = evaluate('ElasticNet', y_test, y_pred_en)
        results.append(r4)

        # ── 5. KNN Regressor (tuned) ──────────────────────────────────────
        print("  [E] KNN Regressor (auto-tuned k)")
        best_k = tune_knn(X_train, y_train)
        knn_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('knn',    KNeighborsRegressor(n_neighbors=best_k,
                                           weights='distance',
                                           metric='euclidean'))
        ])
        knn_pipe.fit(X_train, y_train)
        y_pred_knn_train = knn_pipe.predict(X_train)
        y_pred_knn       = knn_pipe.predict(X_test)
        cv_knn = cross_val_score(knn_pipe, X_train, y_train,
                                 cv=CV_FOLDS, scoring='r2').mean()
        r5 = evaluate(f'KNN (k={best_k})', y_test, y_pred_knn,
                      y_train, y_pred_knn_train)
        r5['CV_R2'] = round(cv_knn, 4)
        results.append(r5)

        all_metrics[target] = results

        # ── Pick best model ───────────────────────────────────────────────
        best_r = max(results, key=lambda x: x['R2'])
        print(f"\n  ★ Best model for {target}: {best_r['model']}  R²={best_r['R2']}")

        # Determine which pipeline won
        pipe_map = {
            'Linear Regression': lr_pipe,
            'Ridge Regression':  ridge_pipe,
            'Lasso Regression':  lasso_pipe,
            'ElasticNet':        en_pipe,
        }
        if best_r['model'].startswith('KNN'):
            best_pipe = knn_pipe
        else:
            best_pipe = pipe_map.get(best_r['model'], lr_pipe)

        best_models[target] = best_pipe

        # ── Save model + feature list ──────────────────────────────────────
        safe_name = target.lower().replace(' ', '_')
        joblib.dump(best_pipe, os.path.join(MODEL_DIR, f'{safe_name}_model.pkl'))
        joblib.dump(features,  os.path.join(MODEL_DIR, f'{safe_name}_features.pkl'))

        # Save Linear Regression and KNN always (for Flask)
        joblib.dump(lr_pipe,   os.path.join(MODEL_DIR, f'{safe_name}_lr.pkl'))
        joblib.dump(knn_pipe,  os.path.join(MODEL_DIR, f'{safe_name}_knn.pkl'))

        # ── Plots ──────────────────────────────────────────────────────────
        plot_model_comparison(target, results)
        plot_actual_vs_predicted(target, y_test, y_pred_lr, y_pred_knn)
        plot_residuals(target, y_test, y_pred_lr, y_pred_knn)
        plot_feature_importance(target, features, lr_pipe)

    # Overall summary
    plot_overall_summary(all_metrics)

    # Save metrics JSON
    metrics_out = {}
    for t, res_list in all_metrics.items():
        metrics_out[t] = res_list
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n[METRICS] Saved to {METRICS_PATH}")

    # Save feature sets
    joblib.dump(feature_sets,
                os.path.join(MODEL_DIR, 'feature_sets.pkl'))

    return all_metrics, best_models, feature_sets


# ─────────────────────────────────────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(all_metrics: dict):
    print("\n" + "="*80)
    print("  FINAL TRAINING SUMMARY")
    print("="*80)
    header = f"{'Target':<22} {'Model':<28} {'MAE':>7} {'RMSE':>7} {'R²':>8} {'Train-R²':>10}"
    print(header)
    print("-"*80)
    for target, results in all_metrics.items():
        best = max(results, key=lambda x: x['R2'])
        tr   = best.get('Train_R2', '-')
        tr_s = f"{tr:.4f}" if isinstance(tr, float) else tr
        print(f"  {target:<20} {best['model']:<28} "
              f"{best['MAE']:>7.4f} {best['RMSE']:>7.4f} "
              f"{best['R2']:>8.4f} {tr_s:>10}")
    print("="*80)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*70)
    print("  STUDENT MENTAL HEALTH — ML TRAINING PIPELINE")
    print("="*70)

    # Step 1 — Preprocess
    print("\n[STEP 1] Preprocessing & Feature Engineering...")
    df = full_preprocess(DATASET_PATH, save_path=PROCESSED_PATH)

    # Step 2 — EDA
    print("\n[STEP 2] Generating EDA Visualizations...")
    generate_all_plots(df)

    # Step 3 — Train
    print("\n[STEP 3] Training Models...")
    all_metrics, best_models, feature_sets = train_all(df)

    # Step 4 — Summary
    print_summary(all_metrics)

    print("\n✅ Training complete!  All models saved to trained_models/")
    print("✅ All graphs saved to static/graphs/")
    print("✅ Processed data saved to dataset/processed_student_data.csv")
