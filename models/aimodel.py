"""
aimodel.py — Healthcare Emergency Prediction System
=====================================================
Ambulance-measurable inputs (same for all 3 models):
    age, sex, heart_rate, blood_pressure,
    spo2, body_temperature, glucose

Usage:  python aimodel.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, "Newdataset")
MODEL_DIR    = BASE_DIR

HEART_CSV       = os.path.join(DATASET_DIR, "heart.csv")
STROKE_CSV      = os.path.join(DATASET_DIR, "stroke.csv")
RESPIRATORY_CSV = os.path.join(DATASET_DIR, "respiratory.csv")

HEART_MODEL_PATH       = os.path.join(MODEL_DIR, "heart_model.pkl")
STROKE_MODEL_PATH      = os.path.join(MODEL_DIR, "stroke_model.pkl")
RESPIRATORY_MODEL_PATH = os.path.join(MODEL_DIR, "respiratory_model.pkl")

FEATURES = ['age', 'sex', 'heart_rate', 'blood_pressure',
            'spo2', 'body_temperature', 'glucose']

# ─────────────────────────────────────────────
# PHYSIOLOGICAL VALID RANGES
# ─────────────────────────────────────────────
VALID_RANGES = {
    'age':              (0,    120,  "years"),
    'heart_rate':       (20,   250,  "bpm"),
    'blood_pressure':   (50,   300,  "mmHg"),
    'spo2':             (50,   100,  "%"),
    'body_temperature': (25.0, 45.0, "degrees C"),
    'glucose':          (20,   1500, "mg/dL"),
}


# ─────────────────────────────────────────────
# CHECK IF MODELS ARE ALREADY SAVED
# ─────────────────────────────────────────────

def models_exist():
    return all(os.path.exists(p) for p in [
        HEART_MODEL_PATH, STROKE_MODEL_PATH, RESPIRATORY_MODEL_PATH])


# ─────────────────────────────────────────────
# INPUT VALIDATION
# ─────────────────────────────────────────────

class InputValidationError(Exception):
    pass


def validate_inputs(age, sex, heart_rate, blood_pressure,
                    spo2, body_temperature, glucose):
    errors        = []
    warnings_list = []

    if isinstance(sex, str):
        sex_lower = sex.strip().lower()
        if sex_lower not in ('male', 'female', 'm', 'f', '0', '1'):
            errors.append(f"sex='{sex}' is invalid. Use 'male' or 'female'.")
            sex_num = None
        else:
            sex_num = 1 if sex_lower in ('male', 'm', '1') else 0
    elif isinstance(sex, (int, float)):
        if sex not in (0, 1):
            errors.append(f"sex={sex} is invalid. Use 1 (male) or 0 (female).")
            sex_num = None
        else:
            sex_num = int(sex)
    else:
        errors.append("sex must be 'male'/'female' or 1/0.")
        sex_num = None

    inputs = {
        'age':              age,
        'heart_rate':       heart_rate,
        'blood_pressure':   blood_pressure,
        'spo2':             spo2,
        'body_temperature': body_temperature,
        'glucose':          glucose,
    }

    cleaned = {}
    for field, value in inputs.items():
        lo, hi, unit = VALID_RANGES[field]

        if value is None or (isinstance(value, float) and np.isnan(value)):
            errors.append(f"{field} is missing. Please provide a value.")
            cleaned[field] = None
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            errors.append(f"{field}='{value}' is not a number.")
            cleaned[field] = None
            continue

        if value < lo:
            errors.append(
                f"{field}={value} is below minimum ({lo} {unit}). "
                f"Please re-check the reading.")
        elif value > hi:
            errors.append(
                f"{field}={value} exceeds maximum ({hi} {unit}). "
                f"Please re-check the reading.")
        else:
            if field == 'age'              and value > 110:
                warnings_list.append(f"age={value} is very high — please confirm.")
            if field == 'heart_rate'       and value < 30:
                warnings_list.append(f"heart_rate={value} bpm is critically low.")
            if field == 'spo2'             and value < 80:
                warnings_list.append(f"spo2={value}% is critically low — verify probe.")
            if field == 'body_temperature' and value > 42.0:
                warnings_list.append(f"temperature={value} C is dangerously high.")
            if field == 'glucose'          and value > 600:
                warnings_list.append(f"glucose={value} mg/dL is critically high — verify.")
            cleaned[field] = value

    if errors:
        raise InputValidationError(
            "\n  INPUT ERRORS FOUND:\n" +
            "\n".join(f"    x {e}" for e in errors))

    return (cleaned['age'], sex_num, cleaned['heart_rate'],
            cleaned['blood_pressure'], cleaned['spo2'],
            cleaned['body_temperature'], cleaned['glucose'],
            warnings_list)


# ─────────────────────────────────────────────
# TRAINING HELPERS
# ─────────────────────────────────────────────

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)


def encode_sex(df):
    if 'sex' in df.columns and df['sex'].dtype == object:
        df['sex'] = df['sex'].str.strip().str.lower().map({'male': 1, 'female': 0})
    return df


def load_dataset(path, target_col):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    df = encode_sex(df)
    X = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    y = df[target_col].apply(pd.to_numeric, errors='coerce')
    return X, y


def build_pipeline(clf):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('clf',     clf),
    ])


def print_metrics(model, X_tr, y_tr, X_te, y_te, name):
    tr_acc = accuracy_score(y_tr, model.predict(X_tr))
    te_acc = accuracy_score(y_te, model.predict(X_te))
    avg    = 'binary' if len(np.unique(y_te)) == 2 else 'weighted'
    yp     = model.predict(X_te)
    prec   = precision_score(y_te, yp, average=avg, zero_division=0)
    rec    = recall_score(y_te,  yp, average=avg, zero_division=0)
    f1     = f1_score(y_te,     yp, average=avg, zero_division=0)
    print(f'\n    [{name}]')
    print(f'      Train Accuracy : {tr_acc*100:.2f}%')
    print(f'      Test  Accuracy : {te_acc*100:.2f}%')
    print(f'      Overfit Gap    : {(tr_acc-te_acc)*100:.2f}%')
    print(f'      Precision      : {prec*100:.2f}%')
    print(f'      Recall         : {rec*100:.2f}%')
    print(f'      F1 Score       : {f1*100:.2f}%')
    return te_acc


RF_PARAMS = dict(
    n_estimators=100, max_depth=8,
    min_samples_split=15, min_samples_leaf=6,
    max_features='sqrt', class_weight='balanced', random_state=42,
)
GB_PARAMS = dict(
    n_estimators=100, max_depth=4,
    min_samples_split=15, min_samples_leaf=6,
    learning_rate=0.08, subsample=0.8,
    max_features='sqrt', random_state=42,
)


def train_best_model(X_tr, y_tr, X_te, y_te, label):
    candidates = {
        'RandomForest':     RandomForestClassifier(**RF_PARAMS),
        'GradientBoosting': GradientBoostingClassifier(**GB_PARAMS),
    }
    print(f'\n{"─"*52}')
    print(f'  Training: {label}')
    print(f'{"─"*52}')
    best_model, best_acc, best_name = None, -1.0, ''
    for name, clf in candidates.items():
        model = build_pipeline(clf)
        model.fit(X_tr, y_tr)
        acc = print_metrics(model, X_tr, y_tr, X_te, y_te, name)
        if acc > best_acc:
            best_acc, best_model, best_name = acc, model, name
    print(f'\n  Best model: {best_name}  (test acc = {best_acc*100:.2f}%)')
    return best_model


def train_all_models():
    ensure_dirs()
    datasets = [
        ('Heart Disease', HEART_CSV,       'target', HEART_MODEL_PATH),
        ('Stroke',        STROKE_CSV,      'stroke', STROKE_MODEL_PATH),
        ('Respiratory',   RESPIRATORY_CSV, 'status', RESPIRATORY_MODEL_PATH),
    ]
    for label, csv_path, target_col, save_path in datasets:
        X, y = load_dataset(csv_path, target_col)
        cv_scores = cross_val_score(
            build_pipeline(RandomForestClassifier(**RF_PARAMS)),
            X, y, cv=5, scoring='accuracy')
        print(f'\n  {label} — 5-fold CV: '
              f'{cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%')
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        model = train_best_model(X_tr, y_tr, X_te, y_te, label)
        joblib.dump(model, save_path)
        print(f'  Saved -> {save_path}')
    print('\n  Training complete. Models saved to disk.')


# ─────────────────────────────────────────────
# PREDICTION  (loads saved models — no retraining)
# ─────────────────────────────────────────────

def load_models():
    return (joblib.load(HEART_MODEL_PATH),
            joblib.load(STROKE_MODEL_PATH),
            joblib.load(RESPIRATORY_MODEL_PATH))


def _prob_positive(model, X):
    proba   = model.predict_proba(X)[0]
    classes = list(model.classes_)
    if 1 in classes:
        return proba[classes.index(1)] * 100
    return proba[-1] * 100


def predict_emergency(age, sex, heart_rate, blood_pressure,
                      spo2, body_temperature, glucose):
    """
    Call this function from frontend/backend with patient vitals.
    Returns a dict with risk percentages and most probable condition.
    Models are loaded from disk — no retraining happens.

    Uses clinical feature weighting to break ties between models
    so the most medically relevant condition is always selected.
    """
    try:
        (age, sex_num, heart_rate, blood_pressure,
         spo2, body_temperature, glucose,
         warnings_list) = validate_inputs(
            age, sex, heart_rate, blood_pressure,
            spo2, body_temperature, glucose)
    except InputValidationError as e:
        print(str(e))
        return None

    if warnings_list:
        print("\n  WARNINGS:")
        for w in warnings_list:
            print(f"    !  {w}")

    heart_m, stroke_m, resp_m = load_models()

    X = np.array([[age, sex_num, heart_rate, blood_pressure,
                   spo2, body_temperature, glucose]])

    # ── Step 1: Raw model probabilities (0–100 each) ─────────
    h_cls      = list(heart_m.classes_)
    heart_pct  = heart_m.predict_proba(X)[0][h_cls.index(1)] * 100 if 1 in h_cls else 0

    s_cls      = list(stroke_m.classes_)
    stroke_pct = stroke_m.predict_proba(X)[0][s_cls.index(1)] * 100 if 1 in s_cls else 0

    r_proba    = resp_m.predict_proba(X)[0]
    r_cls      = list(resp_m.classes_)
    # Class 2 = severe respiratory — the emergency class
    resp_pct   = r_proba[r_cls.index(2)] * 100 if 2 in r_cls else r_proba[-1] * 100

    # ── Step 2: Clinical boost rules ─────────────────────────
    # Each vital boosts ONLY the condition it clinically indicates.
    # Values are independent 0–100 scores — NOT a pie chart.
    # Multiple rules can fire; each is capped at 100.

    # SpO2 — primary respiratory signal
    if spo2 < 85:              resp_pct   = min(resp_pct   + 40, 100)
    elif spo2 < 90:            resp_pct   = min(resp_pct   + 25, 100)
    elif spo2 < 94:            resp_pct   = min(resp_pct   + 10, 100)

    # Body temperature — respiratory signal
    if body_temperature >= 39.5:   resp_pct = min(resp_pct + 20, 100)
    elif body_temperature >= 39.0: resp_pct = min(resp_pct + 12, 100)
    elif body_temperature >= 38.5: resp_pct = min(resp_pct +  6, 100)

    # Heart rate — primary heart signal
# Heart rate — primary heart signal
    # Only boost if OTHER vitals also show signs of distress
    # HR alone in a young healthy patient should not trigger moderate risk
    other_vitals_normal = (spo2 >= 95 and blood_pressure < 150 and
                           body_temperature < 38.5 and glucose < 160)
    if heart_rate > 150:
        heart_pct = min(heart_pct + 40, 100)
    elif heart_rate > 130:
        heart_pct = min(heart_pct + 25, 100)
    elif heart_rate > 110 and not other_vitals_normal:
        heart_pct = min(heart_pct + 15, 100)
    elif heart_rate > 130 and other_vitals_normal:
        heart_pct = min(heart_pct + 10, 100)
    # Blood pressure — primary stroke signal
    if blood_pressure >= 175:  stroke_pct = min(stroke_pct + 35, 100)
    elif blood_pressure >= 160:stroke_pct = min(stroke_pct + 20, 100)
    elif blood_pressure >= 145:stroke_pct = min(stroke_pct +  8, 100)

    # Glucose — secondary stroke signal
    if glucose > 220:          stroke_pct = min(stroke_pct + 20, 100)
    elif glucose > 170:        stroke_pct = min(stroke_pct + 10, 100)

    # Age — stroke amplifier
    if age >= 70:              stroke_pct = min(stroke_pct + 15, 100)
    elif age >= 60:            stroke_pct = min(stroke_pct +  8, 100)

    # Tachycardia override — HR > 140 is a heart emergency signal,
    # not stroke; prevent stroke model from stealing this case
    if heart_rate > 140 and heart_pct > stroke_pct and heart_pct > resp_pct:
        heart_pct = min(heart_pct + 20, 100)

    # ── Step 3: Condition selection ───────────────────────────
    # Each score is an independent risk percentage (0–100).
    # Winner = highest score that clears 70%.
    scores = {
        'Heart Emergency':       heart_pct,
        'Stroke Risk':           stroke_pct,
        'Respiratory Emergency': resp_pct,
    }
    above_thresh = {k: v for k, v in scores.items() if v >= 70}

    if above_thresh:
        condition = max(above_thresh, key=above_thresh.get)
    else:
        top = max([('Heart Disease',       heart_pct),
                   ('Stroke',              stroke_pct),
                   ('Respiratory Problem', resp_pct)],
                  key=lambda t: t[1])
        # All vitals normal check — if nothing is clinically abnormal
        # the condition should always be Low Risk regardless of model score
        all_normal = (
            spo2             >= 95  and
            blood_pressure   <  150 and
            body_temperature <  38.5 and
            glucose          <  160 and
            heart_rate       <  130
        )
        if all_normal:
            condition = 'Low Risk'
        elif top[1] < 40:
            condition = 'Low Risk'
        elif top[1] < 62:
            condition = 'Moderate Risk'
        else:
            condition = top[0]

    return {
        'heart_disease_probability':       round(heart_pct,  2),
        'stroke_probability':              round(stroke_pct, 2),
        'respiratory_problem_probability': round(resp_pct,   2),
        'most_probable_condition':         condition,
    }


# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────

def print_prediction(age, sex, heart_rate, blood_pressure,
                     spo2, body_temperature, glucose, result):
    if result is None:
        print("  Prediction skipped due to invalid inputs.\n")
        return
    print('\n' + '=' * 48)
    print('      AI Emergency Health Prediction')
    print('=' * 48)
    print('\n  Input Vitals (Ambulance-measurable)')
    print(f'  Age              : {age} yrs')
    print(f'  Sex              : {sex}')
    print(f'  Heart Rate       : {heart_rate} bpm')
    print(f'  Blood Pressure   : {blood_pressure} mmHg (systolic)')
    print(f'  SpO2             : {spo2} %')
    print(f'  Body Temperature : {body_temperature} C')
    print(f'  Glucose          : {glucose} mg/dL')
    print('\n  Prediction Results')
    print(f'  Heart Disease Risk  : {result["heart_disease_probability"]:.1f} %')
    print(f'  Stroke Risk         : {result["stroke_probability"]:.1f} %')
    print(f'  Respiratory Risk    : {result["respiratory_problem_probability"]:.1f} %')
    print(f'\n  Most Probable Condition : {result["most_probable_condition"]}')
    print('=' * 48 + '\n')


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print('\n' + '=' * 48)
    print('    Healthcare Emergency Prediction System')
    print('=' * 48)

    # ── Train only on first ever run ──────────────────────────────────
    # Once the 3 .pkl files exist in models/ folder,
    # this block is skipped completely on every future run
    if not models_exist():
        print('\n  First time setup — training models...')
        print('  This will only happen once.\n')
        train_all_models()
    else:
        print('\n  Models loaded from disk. Ready to predict.')

    # ── Keep predicting until user says no ────────────────────────────
    another = True
    while another:
        print('\n' + '-' * 48)
        print('  Enter Patient Vitals')
        print('-' * 48)
        try:
            age  = float(input('\n  Age (years)              : '))
            sex  =       input('  Sex (male / female)      : ').strip()
            hr   = float(input('  Heart Rate (bpm)         : '))
            bp   = float(input('  Blood Pressure (mmHg)    : '))
            spo2 = float(input('  SpO2 (%)                 : '))
            temp = float(input('  Body Temperature (C)     : '))
            gluc = float(input('  Glucose (mg/dL)          : '))
        except ValueError:
            print('\n  Please enter numbers only for numeric fields.\n')
            continue

        result = predict_emergency(age, sex, hr, bp, spo2, temp, gluc)
        print_prediction(age, sex, hr, bp, spo2, temp, gluc, result)

        again = input('  Predict another patient? (yes / no) : ').strip().lower()
        another = again in ('yes', 'y')

    print('\n  Session ended. Goodbye.\n')


if __name__ == '__main__':
    main()