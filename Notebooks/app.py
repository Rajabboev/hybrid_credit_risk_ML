import streamlit as st
import pandas as pd
import numpy as np
import joblib

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

    return (clf, reg, feature_cols,
            data_encoded, data_original,
            numeric_cols, categorical_cols,
            template_row, risk_min, risk_max)

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
    risk_max
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
# TAB 1: Existing client
# =============================
with tab_existing:
    st.subheader("Existing client")

    client_ids = data_original["ID"].unique()
    selected_id = st.selectbox("Choose client ID", sorted(client_ids))

    row_orig = data_original[data_original["ID"] == selected_id].iloc[0]
    row_enc = data_encoded[data_encoded["ID"] == selected_id]
    X_row = row_enc[feature_columns]

    proba_bad = clf_model.predict_proba(X_row)[0][1]
    label = "Bad (High Risk)" if proba_bad > 0.5 else "Good (Low Risk)"
    risk_score_pred = reg_model.predict(X_row)[0]

    true_max_bad = int(row_orig["max_bad"])
    true_risk_score = float(row_orig["risk_score"])
    income = float(row_orig["AMT_INCOME_TOTAL"])

    risk_band, credit_score, recommended_limit = derive_credit_score_and_limit(risk_score_pred, income)

    # Show key info
    st.markdown("### Application summary")
    main_cols = [
        "AMT_INCOME_TOTAL",
        "CNT_CHILDREN",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "CODE_GENDER"
    ]
    summary = row_orig[main_cols].to_frame().rename(columns={0: "Value"})
    st.table(summary)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Classification output")
        st.write(f"**Predicted label:** {label}")
        st.write(f"**Probability of BAD:** {proba_bad:.2%}")
        st.write(f"**True max_bad:** {true_max_bad} ({'Bad' if true_max_bad==1 else 'Good'})")

    with col2:
        st.markdown("#### Regression / risk outputs")
        st.write(f"**Predicted risk_score:** {risk_score_pred:.3f}")
        st.write(f"**True risk_score:** {true_risk_score:.3f}")
        st.write(f"**Risk band:** {risk_band}")
        st.write(f"**Derived credit score (0–1000):** {credit_score:.0f}")
        st.write(f"**Recommended limit (approx.):** {recommended_limit:,.0f}")

# =============================
# TAB 2: New manual application
# =============================
with tab_new:
    st.subheader("New application")

    with st.form("manual_form"):
        st.markdown("##### Personal & employment information")
        col_a, col_b = st.columns(2)
        with col_a:
            age_years = st.number_input("Age (years)", min_value=18, max_value=80, value=30)
            years_employed = st.number_input("Years employed", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
            income = st.number_input("Annual income", min_value=0.0, value=12000000.0, step=100000.0)
            cnt_children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)
        with col_b:
            code_gender = st.selectbox("Gender", ["M", "F"])
            name_income_type = st.selectbox("Income type", sorted(data_original["NAME_INCOME_TYPE"].dropna().unique()))
            name_education_type = st.selectbox("Education", sorted(data_original["NAME_EDUCATION_TYPE"].dropna().unique()))
            name_family_status = st.selectbox("Family status", sorted(data_original["NAME_FAMILY_STATUS"].dropna().unique()))
            name_housing_type = st.selectbox("Housing type", sorted(data_original["NAME_HOUSING_TYPE"].dropna().unique()))

        st.markdown("##### Assets & contact flags")
        col_c, col_d = st.columns(2)
        with col_c:
            flag_own_car = st.selectbox("Own car?", ["N", "Y"])
            flag_own_realty = st.selectbox("Own real estate?", ["N", "Y"])
        with col_d:
            flag_mobil = st.checkbox("Has mobile phone?", value=True)
            flag_work_phone = st.checkbox("Has work phone?", value=False)
            flag_phone = st.checkbox("Has home phone?", value=False)
            flag_email = st.checkbox("Has email?", value=True)

        st.markdown("##### Credit history (self-reported)")
        col_e, col_f = st.columns(2)
        with col_e:
            months_on_book = st.number_input("Months with previous cards/loans", min_value=0, max_value=240, value=12)
        with col_f:
            any_late_str = st.selectbox("Any late payments in history?", ["No", "Yes"])

        submitted = st.form_submit_button("Run risk prediction")

    if submitted:
        # build new row from template
        new_row = template_row.copy()
        new_row["ID"] = -1
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
        new_row["FLAG_MOBIL"] = int(flag_mobil)
        new_row["FLAG_WORK_PHONE"] = int(flag_work_phone)
        new_row["FLAG_PHONE"] = int(flag_phone)
        new_row["FLAG_EMAIL"] = int(flag_email)
        new_row["months_on_book"] = months_on_book
        new_row["any_late"] = 1 if any_late_str == "Yes" else 0

        X_new = encode_single_row(new_row)

        # run models
        proba_bad_new = clf_model.predict_proba(X_new)[0][1]
        label_new = "Bad (High Risk)" if proba_bad_new > 0.5 else "Good (Low Risk)"
        risk_score_pred_new = reg_model.predict(X_new)[0]

        risk_band_new, credit_score_new, recommended_limit_new = derive_credit_score_and_limit(
            risk_score_pred_new, income
        )

        st.markdown("### Prediction for this application")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### Classification")
            st.write(f"**Predicted label:** {label_new}")
            st.write(f"**Probability of BAD:** {proba_bad_new:.2%}")

        with col2:
            st.markdown("#### Risk score (regression)")
            st.write(f"**Predicted risk_score:** {risk_score_pred_new:.3f}")
            st.write(f"**Risk band:** {risk_band_new}")

        with col3:
            st.markdown("#### Derived metrics")
            st.write(f"**Credit score (0–1000):** {credit_score_new:.0f}")
            st.write(f"**Recommended credit limit:** {recommended_limit_new:,.0f}")

        st.info(
            "Note: credit score and recommended limit are heuristic values "
            "derived from the ML risk score and income. They are for demonstration "
            "purposes and not based on a real bank policy."
        )
