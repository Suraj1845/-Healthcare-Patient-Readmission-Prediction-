import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# -----------------------------
# 0) Reproducibility & output
# -----------------------------
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
OUT = Path(".")
OUT.mkdir(exist_ok=True, parents=True)

# -----------------------------
# 1) Generate synthetic dataset
# -----------------------------
N = 6000  # increase for bigger dataset

# Master lists
departments = ["Cardiology", "General Medicine", "Surgery", "Oncology", "Pediatrics", "Orthopedics", "Neurology"]
admission_types = ["Emergency", "Urgent", "Elective"]
discharge_disp = ["Home", "Home with Home-care", "Skilled Nursing Facility", "Rehab", "Against Medical Advice"]
insurance = ["Private", "Medicare/State", "Self-pay"]
primary_dx = [
    "CHF", "COPD", "Pneumonia", "Diabetes", "CKD", "Cancer", "Hip Fracture",
    "Appendicitis", "Stroke", "Sepsis", "Asthma", "Other"
]

# Department -> example doctor names
dept_doctors = {
    "Cardiology": ["Dr. Rao", "Dr. Mehta", "Dr. Kapoor"],
    "General Medicine": ["Dr. Sharma", "Dr. Iyer", "Dr. Banerjee"],
    "Surgery": ["Dr. Verma", "Dr. Singh", "Dr. Khanna"],
    "Oncology": ["Dr. Thomas", "Dr. Nair", "Dr. Mukherjee"],
    "Pediatrics": ["Dr. Patel", "Dr. Das", "Dr. Kulkarni"],
    "Orthopedics": ["Dr. Reddy", "Dr. Gill", "Dr. Bhatia"],
    "Neurology": ["Dr. Menon", "Dr. Roy", "Dr. Dutta"],
}

# Dates within last ~18 months
start_date = datetime.today() - timedelta(days=540)
admit_dates = np.array([start_date + timedelta(days=int(rng.integers(0, 540))) for _ in range(N)])

age = rng.integers(0, 95, N)
sex = rng.choice(["Male", "Female"], size=N)
dept = rng.choice(departments, size=N, p=[0.16, 0.25, 0.18, 0.10, 0.10, 0.12, 0.09])
adm_type = rng.choice(admission_types, size=N, p=[0.55, 0.25, 0.20])
ins = rng.choice(insurance, size=N, p=[0.55, 0.35, 0.10])
dx = rng.choice(primary_dx, size=N)

# Clinical features
prev_adm_6m = rng.poisson(lam=0.6, size=N)  # previous admissions in 6 months
comorb_dm   = rng.choice([0, 1], size=N, p=[0.75, 0.25])  # diabetes
comorb_chf  = rng.choice([0, 1], size=N, p=[0.85, 0.15])  # heart failure
comorb_ckd  = rng.choice([0, 1], size=N, p=[0.88, 0.12])  # kidney disease
comorb_copd = rng.choice([0, 1], size=N, p=[0.86, 0.14])  # lung disease

los = np.clip(np.round(rng.normal(4.5, 3.2, N)), 1, 30).astype(int)  # length of stay (days)
discharge_dates = np.array([admit_dates[i] + timedelta(days=int(los[i])) for i in range(N)])

# Labs/vitals (simplified)
glucose = np.clip(rng.normal(115, 35, N), 60, 400)        # mg/dL
hemoglobin = np.clip(rng.normal(12.8, 2.1, N), 6, 18)     # g/dL
sbp = np.clip(rng.normal(126, 18, N), 80, 200)            # systolic BP
dbp = np.clip(rng.normal(78, 12, N), 45, 120)             # diastolic BP
bmi = np.clip(rng.normal(26.5, 5.5, N), 14, 55)
med_count = np.clip(rng.poisson(lam=8, size=N) + rng.integers(-2, 3, N), 0, 35)
proc_count = np.clip(rng.poisson(lam=1.2, size=N), 0, 8)

# Discharge disposition influenced by age/LOS
disp_probs = []
for i in range(N):
    base = np.array([0.58, 0.14, 0.16, 0.08, 0.04], dtype=float)
    if age[i] >= 70 or los[i] >= 7:
        base += np.array([-0.18, 0.05, 0.10, 0.03, 0.0])
    base = np.clip(base, 0.01, None)
    base = base / base.sum()
    disp_probs.append(base)
disp_idx = [rng.choice(len(discharge_disp), p=p) for p in disp_probs]
disposition = np.array([discharge_disp[i] for i in disp_idx])

# Doctor assignment by department
doctor = np.array([rng.choice(dept_doctors[d]) for d in dept])

# -----------------------------
# 2) Create outcome with signal
# -----------------------------
# Logistic risk score for 30-day readmission (higher = more likely)
logit = -1.2
logit += np.where(age >= 70, 0.45, 0.0)
logit += np.where(prev_adm_6m >= 1, 0.9 + 0.35*(prev_adm_6m-1), 0.0)
logit += np.where(adm_type == "Emergency", 0.35, 0.0)
logit += np.where(disposition == "Skilled Nursing Facility", 0.7, 0.0)
logit += np.where(disposition == "Rehab", 0.35, 0.0)
logit += np.where(comorb_dm == 1, 0.25, 0.0)
logit += np.where(comorb_chf == 1, 0.55, 0.0)
logit += np.where(comorb_ckd == 1, 0.35, 0.0)
logit += np.where(comorb_copd == 1, 0.30, 0.0)
logit += np.where(glucose >= 180, 0.35, 0.0)
logit += np.where(hemoglobin < 10, 0.25, 0.0)
logit += np.where(los >= 6, 0.25, 0.0)
logit += np.where(bmi >= 35, 0.15, 0.0)
logit += np.where(dept == "Oncology", 0.25, 0.0)
logit += rng.normal(0, 0.35, N)

prob = 1 / (1 + np.exp(-logit))
readmit = np.where(rng.random(N) < prob, "Yes", "No")

# -----------------------------
# 3) Assemble DataFrame
# -----------------------------
df = pd.DataFrame({
    "patient_id": [f"PT{100000+i}" for i in range(N)],
    "admit_date": admit_dates,
    "discharge_date": discharge_dates,
    "age": age,
    "sex": sex,
    "department": dept,
    "doctor": doctor,
    "admission_type": adm_type,
    "insurance": ins,
    "primary_diagnosis": dx,
    "prev_admissions_6m": prev_adm_6m,
    "length_of_stay": los,
    "glucose": np.round(glucose, 1),
    "hemoglobin": np.round(hemoglobin, 1),
    "sbp": np.round(sbp, 0),
    "dbp": np.round(dbp, 0),
    "bmi": np.round(bmi, 1),
    "med_count": med_count,
    "procedure_count": proc_count,
    "discharge_disposition": disposition,
    "readmitted_30d": readmit
})

# Save dataset
dataset_path = OUT / "readmission_dataset.csv"
df.to_csv(dataset_path, index=False)

# -----------------------------
# 4) Train/test split + preprocessing
# -----------------------------
TARGET = "readmitted_30d"
ID_COLS = ["patient_id", "admit_date", "discharge_date"]

X = df.drop(columns=[TARGET] + ID_COLS)
y = (df[TARGET] == "Yes").astype(int)

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ]
)

# Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1500),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced_subsample"
    ),
}

# Try XGBoost; fallback to GradientBoosting
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=RANDOM_STATE
    )
except Exception:
    models["GradientBoosting (fallback)"] = GradientBoostingClassifier(random_state=RANDOM_STATE)

# -----------------------------
# 5) Train/evaluate
# -----------------------------
trained = {}
rows = []
for name, model in models.items():
    pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    rows.append({
        "model": name,
        "roc_auc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0)
    })
    trained[name] = pipe

metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
metrics.to_csv(OUT / "model_metrics.csv", index=False)

best_name = metrics.iloc[0]["model"]
best_model = trained[best_name]
print(f"Best model by ROC-AUC: {best_name}")

# -----------------------------
# 6) Score ALL patients + export
# -----------------------------
all_proba = best_model.predict_proba(X)[:, 1]
all_pred = (all_proba >= 0.5).astype(int)

pred_df = pd.DataFrame({
    "patient_id": df["patient_id"],
    "readmit_prob": np.round(all_proba, 6),
    "readmit_predicted": all_pred
})

# Merge with original for Power BI
pbi_export = df.merge(pred_df, on="patient_id", how="left")

pred_df.to_csv(OUT / "readmission_predictions.csv", index=False)
pbi_export.to_csv(OUT / "powerbi_readmission_export.csv", index=False)

# -----------------------------
# 7) Feature importance (top 25)
# -----------------------------
# Get post-encoding feature names
ohe = best_model.named_steps["prep"].named_transformers_["cat"]
cat_features = list(ohe.get_feature_names_out(cat_cols))
feature_names = cat_features + num_cols

model_final = best_model.named_steps["model"]
if hasattr(model_final, "feature_importances_"):
    importances = model_final.feature_importances_
elif hasattr(model_final, "coef_"):
    importances = np.abs(np.ravel(model_final.coef_))
else:
    importances = np.full(len(feature_names), np.nan)

imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}) \
           .sort_values("importance", ascending=False) \
           .head(25)
imp_df.to_csv(OUT / "feature_importance_top25.csv", index=False)

# -----------------------------
# 8) Helpful aggregations for PBI
# -----------------------------
# Department trends (avg prob and rate)
dept_agg = pbi_export.groupby("department").agg(
    patients=("patient_id", "nunique"),
    avg_readmit_prob=("readmit_prob", "mean"),
    readmit_rate=("readmit_predicted", "mean")
).reset_index()
dept_agg.to_csv(OUT / "department_summary.csv", index=False)

# Doctor performance
doc_agg = pbi_export.groupby(["department", "doctor"]).agg(
    patients=("patient_id", "nunique"),
    avg_readmit_prob=("readmit_prob", "mean"),
    readmit_rate=("readmit_predicted", "mean")
).reset_index()
doc_agg.to_csv(OUT / "doctor_summary.csv", index=False)

print("✅ Files saved to current folder:")
print(" - readmission_dataset.csv")
print(" - readmission_predictions.csv")
print(" - powerbi_readmission_export.csv")
print(" - model_metrics.csv")
print(" - feature_importance_top25.csv")
print(" - department_summary.csv")
print(" - doctor_summary.csv")









