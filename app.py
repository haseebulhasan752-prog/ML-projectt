"""
=============================================================================
Student Mental Health & Academic Performance Prediction System
Flask Web Application
=============================================================================
"""

import os, json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

app  = Flask(__name__)
BASE = os.path.dirname(__file__)

# ── Load feature sets and models ──────────────────────────────────────────────
MODEL_DIR    = os.path.join(BASE, 'trained_models')
METRICS_PATH = os.path.join(BASE, 'static', 'model_metrics.json')

TARGETS   = ['CGPA', 'Stress_Level', 'Depression_Score', 'Anxiety_Score']
SAFE_NAMES = {t: t.lower().replace(' ', '_') for t in TARGETS}

feature_sets = joblib.load(os.path.join(MODEL_DIR, 'feature_sets.pkl'))

# Load best model, LR, and KNN for each target
models = {}
for t in TARGETS:
    sn = SAFE_NAMES[t]
    models[t] = {
        'best': joblib.load(os.path.join(MODEL_DIR, f'{sn}_model.pkl')),
        'lr':   joblib.load(os.path.join(MODEL_DIR, f'{sn}_lr.pkl')),
        'knn':  joblib.load(os.path.join(MODEL_DIR, f'{sn}_knn.pkl')),
    }

with open(METRICS_PATH) as f:
    all_metrics = json.load(f)

# ── Input field definitions (for the dynamic form) ────────────────────────────
INPUT_FIELDS = {
    'Age':                      {'type': 'number', 'min': 17, 'max': 45, 'step': 1,   'default': 21},
    'Gender':                   {'type': 'select', 'options': ['Male', 'Female'],      'default': 'Male'},
    'CGPA':                     {'type': 'number', 'min': 0.0, 'max': 4.0, 'step': 0.01,'default': 3.0},
    'Stress_Level':             {'type': 'number', 'min': 0, 'max': 10, 'step': 1,    'default': 5},
    'Depression_Score':         {'type': 'number', 'min': 0, 'max': 10, 'step': 1,    'default': 4},
    'Anxiety_Score':            {'type': 'number', 'min': 0, 'max': 10, 'step': 1,    'default': 4},
    'Sleep_Quality':            {'type': 'select', 'options': ['Poor','Average','Good'],'default': 'Average'},
    'Physical_Activity':        {'type': 'select', 'options': ['Low','Moderate','High'],'default': 'Moderate'},
    'Diet_Quality':             {'type': 'select', 'options': ['Poor','Average','Good'],'default': 'Average'},
    'Social_Support':           {'type': 'select', 'options': ['Low','Moderate','High'],'default': 'Moderate'},
    'Substance_Use':            {'type': 'select', 'options': ['Never','Occasionally','Frequently'],'default': 'Never'},
    'Counseling_Service_Use':   {'type': 'select', 'options': ['Never','Occasionally','Frequently'],'default': 'Never'},
    'Family_History':           {'type': 'select', 'options': ['No','Yes'],            'default': 'No'},
    'Chronic_Illness':          {'type': 'select', 'options': ['No','Yes'],            'default': 'No'},
    'Financial_Stress':         {'type': 'number', 'min': 0, 'max': 5, 'step': 1,     'default': 2},
    'Extracurricular_Involvement':{'type': 'select','options': ['Low','Moderate','High'],'default': 'Moderate'},
    'Semester_Credit_Load':     {'type': 'number', 'min': 15, 'max': 29, 'step': 1,   'default': 20},
    'Course':                   {'type': 'select', 'options': ['Business','Computer Science',
                                                               'Engineering','Law','Medicine',
                                                               'Others','Psychology'],  'default': 'Engineering'},
    'Relationship_Status':      {'type': 'select', 'options': ['Single','In a Relationship','Married'],
                                                                                        'default': 'Single'},
    'Residence_Type':           {'type': 'select', 'options': ['On-Campus','Off-Campus','With Family'],
                                                                                        'default': 'On-Campus'},
}

# ── Ordinal / binary maps (mirror preprocess.py) ─────────────────────────────
ORDINAL_MAPS = {
    'Sleep_Quality':              {'Poor': 0, 'Average': 1, 'Good': 2},
    'Physical_Activity':          {'Low': 0, 'Moderate': 1, 'High': 2},
    'Diet_Quality':               {'Poor': 0, 'Average': 1, 'Good': 2},
    'Social_Support':             {'Low': 0, 'Moderate': 1, 'High': 2},
    'Extracurricular_Involvement':{'Low': 0, 'Moderate': 1, 'High': 2},
    'Counseling_Service_Use':     {'Never': 0, 'Occasionally': 1, 'Frequently': 2},
    'Substance_Use':              {'Never': 0, 'Occasionally': 1, 'Frequently': 2},
}
BINARY_MAPS = {
    'Family_History':  {'No': 0, 'Yes': 1},
    'Chronic_Illness': {'No': 0, 'Yes': 1},
    'Gender':          {'Male': 0, 'Female': 1},
}

# Reference processed data for percentile / mean CGPA
processed_df = pd.read_csv(os.path.join(BASE, 'dataset', 'processed_student_data.csv'))
CGPA_MEAN    = processed_df['CGPA'].mean()


def build_feature_row(form: dict, target: str) -> pd.DataFrame:
    """
    Convert form inputs into the engineered feature vector
    that the trained model expects.
    """
    raw = {}

    # ── Raw numeric fields ────────────────────────────────────────────────
    raw['Age']                   = float(form.get('Age', 21))
    raw['CGPA']                  = float(form.get('CGPA', 3.0))
    raw['Stress_Level']          = float(form.get('Stress_Level', 5))
    raw['Depression_Score']      = float(form.get('Depression_Score', 4))
    raw['Anxiety_Score']         = float(form.get('Anxiety_Score', 4))
    raw['Financial_Stress']      = float(form.get('Financial_Stress', 2))
    raw['Semester_Credit_Load']  = float(form.get('Semester_Credit_Load', 20))

    # ── Ordinal encoding ──────────────────────────────────────────────────
    for col, mapping in ORDINAL_MAPS.items():
        val = form.get(col, list(mapping.keys())[1])
        raw[col] = mapping.get(str(val), 1)

    # ── Binary encoding ───────────────────────────────────────────────────
    for col, mapping in BINARY_MAPS.items():
        val = form.get(col, 'No')
        raw[col] = mapping.get(str(val), 0)

    # ── One-hot encoding (Course, Relationship_Status, Residence_Type) ───
    for col, options_key in [('Course', 'Course'),
                              ('Relationship_Status', 'Relationship_Status'),
                              ('Residence_Type', 'Residence_Type')]:
        selected = form.get(col, INPUT_FIELDS[col]['default'])
        all_options = INPUT_FIELDS[col]['options']
        # drop_first=True means skip the first option
        for opt in all_options[1:]:
            raw[f'{col}_{opt}'] = 1 if selected == opt else 0

    # ── Feature engineering (mirror preprocess.py) ────────────────────────
    sq  = raw['Sleep_Quality']
    pa  = raw['Physical_Activity']
    dq  = raw['Diet_Quality']
    ss  = raw['Social_Support']
    sl  = raw['Stress_Level']
    ds  = raw['Depression_Score']
    ans = raw['Anxiety_Score']
    fs  = raw['Financial_Stress']
    cgpa = raw['CGPA']
    ci  = raw['Chronic_Illness']
    fh  = raw['Family_History']
    su  = raw['Substance_Use']
    cu  = raw['Counseling_Service_Use']
    scl = raw['Semester_Credit_Load']
    age = raw['Age']

    raw['Wellness_Score']            = (sq + pa + dq) / 3.0
    raw['Lifestyle_Index']           = sq*0.4 + pa*0.35 + dq*0.25
    raw['Wellbeing_vs_Burden']       = raw['Wellness_Score'] - (sl / 10.0)
    raw['Mental_Burden']             = (sl + ds + ans) / 3.0
    raw['Support_Deficit']           = (10 - ss*(10/2)) / 10.0
    raw['Academic_Pressure']         = (scl * sl) / (29 * 10)
    raw['Sleep_x_Activity']          = sq * pa
    raw['Stress_x_Financial']        = sl * fs
    raw['Depression_x_Sleep']        = ds * sq
    raw['Anxiety_x_Support']         = ans * ss
    raw['Chronic_Financial']         = ci * fs
    raw['Age_Credit_interaction']    = age * scl
    raw['CGPA_stress_interaction']   = cgpa * sl
    raw['Risk_Score']                = ds*0.3 + ans*0.3 + sl*0.2 + fs*0.1 + ci*0.1
    raw['Family_History_Risk']       = fh * raw['Risk_Score']
    raw['Substance_Counseling_ratio']= su / (cu + 1)
    raw['Total_Mental_Load']         = sl + ds + ans
    raw['Avg_Mental_Score']          = raw['Total_Mental_Load'] / 3.0

    for col_sq, val_sq in [('Sleep_Quality', sq), ('Physical_Activity', pa),
                            ('Diet_Quality', dq), ('Social_Support', ss)]:
        raw[f'{col_sq}_sq'] = val_sq ** 2

    raw['CGPA_deviation']  = cgpa - CGPA_MEAN
    raw['CGPA_percentile'] = float(
        (processed_df['CGPA'] < cgpa).sum() / len(processed_df) * 100
    )

    # ── Build DataFrame with exactly the features the model expects ───────
    feats = feature_sets[target]
    row   = {}
    for f in feats:
        row[f] = raw.get(f, 0.0)

    return pd.DataFrame([row]).values


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html',
                           targets=TARGETS,
                           fields=INPUT_FIELDS)


@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    target    = form_data.get('target', 'CGPA')

    if target not in TARGETS:
        return jsonify({'error': 'Invalid target'}), 400

    try:
        X = build_feature_row(form_data, target)

        pred_lr   = float(models[target]['lr'].predict(X)[0])
        pred_knn  = float(models[target]['knn'].predict(X)[0])
        pred_best = float(models[target]['best'].predict(X)[0])

        # Clamp to realistic ranges
        clamp_map = {
            'CGPA':             (0.0, 4.0),
            'Stress_Level':     (0.0, 10.0),
            'Depression_Score': (0.0, 10.0),
            'Anxiety_Score':    (0.0, 10.0),
        }
        lo, hi = clamp_map[target]
        pred_lr   = round(max(lo, min(hi, pred_lr)),   2)
        pred_knn  = round(max(lo, min(hi, pred_knn)),  2)
        pred_best = round(max(lo, min(hi, pred_best)), 2)

        # Metrics for display
        target_metrics = {m['model']: m for m in all_metrics.get(target, [])}
        lr_r2  = target_metrics.get('Linear Regression', {}).get('R2', 'N/A')
        knn_r2 = max((v.get('R2', 0) for k, v in target_metrics.items()
                      if k.startswith('KNN')), default='N/A')

        # Confidence bucket
        best_r2 = max((m['R2'] for m in all_metrics.get(target, [])), default=0)
        if best_r2 >= 0.9:
            confidence, conf_class = 'High (≥90%)', 'success'
        elif best_r2 >= 0.8:
            confidence, conf_class = 'Good (≥80%)', 'info'
        else:
            confidence, conf_class = 'Moderate', 'warning'

        return render_template(
            'result.html',
            target=target,
            pred_best=pred_best,
            pred_lr=pred_lr,
            pred_knn=pred_knn,
            confidence=confidence,
            conf_class=conf_class,
            lr_r2=lr_r2,
            knn_r2=knn_r2 if isinstance(knn_r2, float) else 'N/A',
            form_data=form_data,
            graph_lr=f'model_comparison_{target}.png',
            graph_avp=f'actual_vs_predicted_{target}.png',
            graph_feat=f'feature_importance_{target}.png',
        )

    except Exception as e:
        return render_template('predict.html',
                               targets=TARGETS,
                               fields=INPUT_FIELDS,
                               error=str(e))


@app.route('/dashboard')
def dashboard():
    graphs = [f for f in os.listdir(os.path.join(BASE, 'static', 'graphs'))
              if f.endswith('.png')]
    metrics_summary = {}
    for t, results in all_metrics.items():
        best = max(results, key=lambda x: x['R2'])
        metrics_summary[t] = best
    return render_template('dashboard.html',
                           graphs=graphs,
                           metrics=metrics_summary,
                           all_metrics=all_metrics)


@app.route('/api/metrics')
def api_metrics():
    return jsonify(all_metrics)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
