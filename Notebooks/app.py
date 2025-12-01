import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# =============================
# Load models and reference data
# =============================
@st.cache_resource
def load_models_and_data():
    clf = joblib.load("models/xgb_clf_tuned.joblib")
    reg = joblib.load("models/xgb_reg_tuned.joblib")
    feature_cols = joblib.load("models/feature_columns.joblib")
    data_encoded = pd.read_csv("data/data_encoded.csv")
    data_original = pd.read_csv("data/data_original.csv")

    # work out numeric / categorical from original data
    numeric_cols = data_original.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["max_bad", "risk_score"]]
    categorical_cols = data_original.select_dtypes(include=["object"]).columns.tolist()

    num_medians = data_original[numeric_cols].median()

    # template row with medians / "Unknown"
    template_row = pd.DataFrame(columns=data_original.columns, index=[0])
    template_row[numeric_cols] = num_medians
    template_row[categorical_cols] = "Unknown"
    template_row["ID"] = -1
    template_row["max_bad"] = 0
    template_row["risk_score"] = 0.0

    risk_min = data_original["risk_score"].min()
    risk_max = data_original["risk_score"].max()

    # --------- GLOBAL MODEL METRICS (for dashboard) ----------
    X_all = data_encoded[feature_cols]
    y_all = data_original["max_bad"]
    risk_true = data_original["risk_score"]

    # classification metrics
    y_proba_all = clf.predict_proba(X_all)[:, 1]
    y_pred_all = (y_proba_all > 0.5).astype(int)

    clf_metrics = {
        "accuracy": accuracy_score(y_all, y_pred_all),
        "precision": precision_score(y_all, y_pred_all, zero_division=0),
        "recall": recall_score(y_all, y_pred_all),
        "f1": f1_score(y_all, y_pred_all),
        "auc": roc_auc_score(y_all, y_proba_all),
    }

    # regression metrics
    risk_pred_all = reg.predict(X_all)
    reg_metrics = {
        "mse": mean_squared_error(risk_true, risk_pred_all),
        "rmse": np.sqrt(mean_squared_error(risk_true, risk_pred_all)),
        "mae": mean_absolute_error(risk_true, risk_pred_all),
        "r2": r2_score(risk_true, risk_pred_all),
    }

    return (
        clf,
        reg,
        feature_cols,
        data_encoded,
        data_original,
        numeric_cols,
        categorical_cols,
        template_row,
        risk_min,
        risk_max,
        clf_metrics,
        reg_metrics,
    )

(
    clf_model,
    reg_model,
    feature_columns,
    data_encoded,
    data_original,
    numeric_cols,
    categorical_cols,
    template_row,
    risk_min,
    risk_max,
    clf_metrics,
    reg_metrics,
) = load_models_and_data()


# =============================
# Helper: encode a single row
# =============================
def encode_single_row(row_orig: pd.DataFrame) -> pd.DataFrame:
    """Apply same preprocessing + one-hot encoding as training, then
    reindex to model's feature_columns."""
    # fill missing
    row_orig = row_orig.copy()
    row_orig[numeric_cols] = row_orig[numeric_cols].fillna(template_row[numeric_cols].iloc[0])
    row_orig[categorical_cols] = row_orig[categorical_cols].fillna("Unknown")

    # one-hot encode
    row_enc = pd.get_dummies(row_orig, columns=categorical_cols, drop_first=True)

    # align columns with training features
    row_enc = row_enc.reindex(columns=feature_columns, fill_value=0)

    return row_enc

# =============================
# Helper: derive credit score & limit
# =============================
def derive_credit_score_and_limit(risk_score_pred: float, income: float):
    # normalise risk to 0–1
    if risk_max > risk_min:
        risk_norm = (risk_score_pred - risk_min) / (risk_max - risk_min)
    else:
        risk_norm = 0.5
    risk_norm = float(np.clip(risk_norm, 0, 1))

    # convert to 0–1000 credit score (higher = better)
    credit_score = (1.0 - risk_norm) * 1000.0

    # simple heuristic for recommended credit limit based on risk
    if risk_norm < 0.3:
        risk_band = "Low"
        limit_multiplier = 0.5   # up to 50% of annual income
    elif risk_norm < 0.6:
        risk_band = "Medium"
        limit_multiplier = 0.3   # up to 30%
    else:
        risk_band = "High"
        limit_multiplier = 0.1   # up to 10%

    recommended_limit = income * limit_multiplier

    return risk_band, credit_score, recommended_limit

def describe_default_probability(prob: float) -> str:
    """
    Turn raw probability into a user-friendly explanation.
    """
    if prob < 0.10:
        band = "Low"
        msg = "The model expects this type of client to very rarely become seriously delinquent."
    elif prob < 0.30:
        band = "Medium"
        msg = "There is a noticeable chance of serious delinquency. Case should be reviewed carefully."
    else:
        band = "High"
        msg = "This client is very likely to become seriously delinquent compared to others."

    return f"Risk level: **{band}** (about {prob:.1%} chance of serious delinquency).\n\n{msg}"


# =============================
# Streamlit UI
# =============================
st.title("Hybrid Credit Risk Prediction App")

st.markdown(
"""
This app uses a **tuned XGBoost classifier** (Good/Bad) and a **regression model**
to produce a **numeric risk score** and an additional **credit score (0–1000)**.

You can either:
- Select an **existing client** from the dataset, or  
- Enter a **new application** manually.
"""
)

tab_existing, tab_new = st.tabs(["Existing client from dataset", "New application (manual input)"])
# =============================
# TAB 1: Existing client (dashboard view)
# =============================
with tab_existing:
    st.subheader("Existing client from training dataset")



    st.markdown("### Select client")

    client_ids = data_original["ID"].unique()
    selected_id = st.selectbox("Client ID", sorted(client_ids))

    row_orig = data_original[data_original["ID"] == selected_id].iloc[0]
    row_enc = data_encoded[data_encoded["ID"] == selected_id]
    X_row = row_enc[feature_columns]

    # predictions
    proba_bad = clf_model.predict_proba(X_row)[0][1]
    label = "Bad (High Risk)" if proba_bad > 0.5 else "Good (Low Risk)"
    risk_score_pred = reg_model.predict(X_row)[0]

    true_max_bad = int(row_orig["max_bad"])
    true_risk_score = float(row_orig["risk_score"])
    income = float(row_orig["AMT_INCOME_TOTAL"])

    risk_band, credit_score, recommended_limit = derive_credit_score_and_limit(
        risk_score_pred, income
    )

    # ----- result dashboard -----
    st.markdown("### Result for selected client")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        # Smaller and nicer label badge
        if "Bad" in label:
            color = "#ff4d4d"  # red
        else:
            color = "#4CAF50"  # green

        st.markdown(
            f"""
            <div style="
                background-color:{color};
                padding:8px 12px;
                border-radius:6px;
                text-align:center;
                font-size:16px;
                font-weight:600;
                color:white;
                margin-bottom:8px;">
                {label}
            </div>
            """,
            unsafe_allow_html=True
        )
    with m2:
        st.metric("Default probability", f"{proba_bad:.1%}")
    with m3:
        st.metric("Risk score", f"{risk_score_pred:.3f}")
    with m4:
        st.metric("Derived credit score", f"{credit_score:.0f}")

    st.markdown(describe_default_probability(proba_bad))

    st.markdown("---")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### Application summary")
        main_cols = [
            "AMT_INCOME_TOTAL",
            "CNT_CHILDREN",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "CODE_GENDER",
        ]
        summary = row_orig[main_cols].to_frame().rename(columns={0: "Value"})
        st.table(summary)

    with colB:
        st.markdown("#### True historical outcome & recommendation")

        st.write(f"**True historical label (max_bad):** {true_max_bad} "
                 f"({'Bad' if true_max_bad == 1 else 'Good'})")
        st.write(f"**True historical risk_score:** {true_risk_score:.3f}")
        st.write(f"**Risk band (from regression):** {risk_band}")
        st.write(f"**Recommended credit limit (demo rule):** {recommended_limit:,.0f}")

        st.caption(
            "The true labels are taken from the historical dataset. "
            "In a live system, only the model predictions would be available, "
            "and business rules would map them to approve / manual review / reject decisions."
        )

        # ----- model performance cards -----
    st.markdown("### Model performance on historical data")

    c_acc, c_rec, c_auc, c_r2 = st.columns(4)
    with c_acc:
        st.metric("Accuracy", f"{clf_metrics['accuracy']:.1%}")
    with c_rec:
        st.metric("Recall (catching BAD)", f"{clf_metrics['recall']:.1%}")
    with c_auc:
        st.metric("ROC–AUC", f"{clf_metrics['auc']:.2f}")
    with c_r2:
        st.metric("Risk R² (regression)", f"{reg_metrics['r2']:.2f}")

    st.caption(
        "Classification metrics are calculated on the full historical dataset. "
        "Recall and AUC are especially important in credit risk, because missing "
        "risky clients is more costly than declining a few good ones."
    )

# =============================
# TAB 2: New manual application (simplified)
# =============================
with tab_new:
    st.subheader("New application (simplified)")

    st.markdown(
        "Provide a few key details about the applicant. "
        "Less important technical fields are filled with typical values from the training data."
    )

    with st.form("manual_form"):

        # ------- Personal & income profile -------
        st.markdown("#### 1. Personal & income profile")
        c1, c2 = st.columns(2)

        with c1:
            age_years = st.number_input(
                "Age (years)", min_value=18, max_value=80, value=30, step=1
            )
            years_employed = st.number_input(
                "Years employed", min_value=0.0, max_value=50.0, value=3.0, step=0.5
            )
            income = st.number_input(
                "Annual income", min_value=0.0, value=12_000_000.0, step=100_000.0
            )

        with c2:
            cnt_children = st.number_input(
                "Number of children", min_value=0, max_value=10, value=0, step=1
            )
            code_gender = st.radio("Gender", ["M", "F"], horizontal=True)
            name_family_status = st.selectbox(
                "Family status",
                sorted(data_original["NAME_FAMILY_STATUS"].dropna().unique())
            )

        # ------- Socio-economic segment -------
        st.markdown("#### 2. Socio-economic segment")
        c3, c4 = st.columns(2)

        with c3:
            name_income_type = st.selectbox(
                "Income type",
                sorted(data_original["NAME_INCOME_TYPE"].dropna().unique())
            )
            name_education_type = st.selectbox(
                "Education",
                sorted(data_original["NAME_EDUCATION_TYPE"].dropna().unique())
            )

        with c4:
            name_housing_type = st.selectbox(
                "Housing type",
                sorted(data_original["NAME_HOUSING_TYPE"].dropna().unique())
            )
            flag_own_realty = st.selectbox("Own real estate?", ["N", "Y"])
            flag_own_car = st.selectbox("Own car?", ["N", "Y"])

        # ------- Credit behaviour -------
        st.markdown("#### 3. Credit behaviour (self-reported)")
        c5, c6 = st.columns(2)

        with c5:
            months_on_book = st.number_input(
                "Months with previous cards/loans",
                min_value=0, max_value=240, value=12
            )

        with c6:
            any_late_str = st.radio(
                "Any late payments in history?",
                ["No", "Yes"], horizontal=True
            )

        submitted = st.form_submit_button("Run risk prediction")

    if submitted:
        # build new row starting from template defaults
        new_row = template_row.copy()
        new_row["ID"] = -1

        # map inputs to original feature space
        new_row["AMT_INCOME_TOTAL"] = income
        new_row["CNT_CHILDREN"] = cnt_children
        new_row["DAYS_BIRTH"] = -int(age_years * 365)        # dataset uses negative days
        new_row["DAYS_EMPLOYED"] = -int(years_employed * 365)

        new_row["CODE_GENDER"] = code_gender
        new_row["NAME_INCOME_TYPE"] = name_income_type
        new_row["NAME_EDUCATION_TYPE"] = name_education_type
        new_row["NAME_FAMILY_STATUS"] = name_family_status
        new_row["NAME_HOUSING_TYPE"] = name_housing_type

        new_row["FLAG_OWN_CAR"] = flag_own_car
        new_row["FLAG_OWN_REALTY"] = flag_own_realty

        # keep contact flags simple defaults (not shown in UI)
        new_row["FLAG_MOBIL"] = 1
        new_row["FLAG_WORK_PHONE"] = template_row["FLAG_WORK_PHONE"].iloc[0]
        new_row["FLAG_PHONE"] = template_row["FLAG_PHONE"].iloc[0]
        new_row["FLAG_EMAIL"] = template_row["FLAG_EMAIL"].iloc[0]

        # behavioural features
        new_row["months_on_book"] = months_on_book
        new_row["any_late"] = 1 if any_late_str == "Yes" else 0

        # encode and predict
        X_new = encode_single_row(new_row)
        proba_bad_new = clf_model.predict_proba(X_new)[0][1]
        label_new = "Bad (High Risk)" if proba_bad_new > 0.5 else "Good (Low Risk)"
        risk_score_pred_new = reg_model.predict(X_new)[0]

        risk_band_new, credit_score_new, recommended_limit_new = derive_credit_score_and_limit(
            risk_score_pred_new, income
        )

        # ------- nicer UI for results -------
        st.markdown("### Prediction summary")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Probability of BAD", f"{proba_bad_new:.1%}")
        with m2:
            st.metric("Risk score", f"{risk_score_pred_new:.3f}")
        with m3:
            st.metric("Credit score (0–1000)", f"{credit_score_new:.0f}")
        with m4:
            st.metric("Risk band", risk_band_new)

        st.markdown("### Recommended decision support")

        st.write(f"**Predicted label:** {label_new}")
        st.write(f"**Recommended credit limit:** {recommended_limit_new:,.0f}")

        st.info(
            "This decision is based on a hybrid ML model: a classifier for BAD/GOOD "
            "and a regression model for continuous risk_score. Less important features "
            "are fixed at typical values, so the focus is on the key drivers such as "
            "age, income, employment, socio-economic segment and past credit behaviour."
        )