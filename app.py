from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "churndb.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.pkl"


def get_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # Simple, explainable feature engineering to match training pipeline
    # tenure_band
    if "tenure" in df.columns:
        df["tenure_band"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72, np.inf], labels=["0-12", "13-24", "25-48", "49-72", "73+"])
    # charges_per_month
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        def _charges_per_month(r):
            try:
                if pd.notna(r["TotalCharges"]) and r.get("tenure") and r["tenure"] > 0:
                    return r["TotalCharges"] / r["tenure"]
            except Exception:
                pass
            return r.get("MonthlyCharges")

        df["charges_per_month"] = df.apply(_charges_per_month, axis=1)
    # num_services: count of positive service flags
    service_cols = [
        c for c in [
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        if c in df.columns
    ]
    if service_cols:
        df["num_services"] = df[service_cols].apply(lambda col: col.map({"Yes": 1, "No": 0}).fillna(0)).sum(axis=1)
    return df


def get_model():
    if not MODEL_PATH.exists():
        from train_model import build_pipeline

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model = build_pipeline()
        joblib.dump(model, MODEL_PATH)
    return joblib.load(MODEL_PATH)


def median_monthly_charge(df: pd.DataFrame) -> float:
    return float(df["MonthlyCharges"].median())


def assign_state(customer: pd.Series, churn_probability: float, df: pd.DataFrame) -> tuple[str, list[str]]:
    contract = customer["Contract"]
    tenure = int(customer["tenure"])
    monthly = float(customer["MonthlyCharges"])
    total = float(customer["TotalCharges"])
    internet = customer["InternetService"]
    payment = customer["PaymentMethod"]
    senior = int(customer["SeniorCitizen"])
    partner = customer["Partner"]
    dependents = customer["Dependents"]
    median_monthly = median_monthly_charge(df)

    reasons: list[str] = []

    if churn_probability >= 0.75:
        if contract == "Month-to-month":
            reasons.append(f"The customer is on a {contract} plan, which is a strong churn-risk pattern in this dataset.")
        if tenure <= 12:
            reasons.append(f"Tenure is only {tenure} months, which indicates the relationship is still in its early stage.")
        if monthly >= median_monthly:
            reasons.append(f"Monthly charges are {monthly:.2f} DZD, above the portfolio median of {median_monthly:.2f} DZD.")
        if payment in ["Electronic check", "Mailed check"]:
            reasons.append(f"The payment method '{payment}' is typically associated with weaker retention patterns.")
        if not reasons:
            reasons.append("The model assigns a very high churn probability based on the customer profile.")
        return "Left", reasons

    if churn_probability >= 0.5:
        if contract == "Month-to-month":
            reasons.append(f"The customer is on a {contract} contract, which increases churn risk compared with longer plans.")
        if tenure <= 18:
            reasons.append(f"Tenure is {tenure} months, suggesting the customer is still relatively new and more price-sensitive.")
        if internet == "Fiber optic":
            reasons.append("Fiber optic service is often linked with higher sensitivity to competitor offers in this customer segment.")
        if payment in ["Electronic check", "Mailed check"]:
            reasons.append(f"Payment method '{payment}' is a weaker retention signal than automatic payment options.")
        if not reasons:
            reasons.append("The model indicates a moderate churn risk based on this profile.")
        return "May Leave", reasons

    if churn_probability >= 0.35:
        reasons.append("The model places this customer in a watchlist zone: risk is present but not yet critical.")
        if contract != "Two year":
            reasons.append(f"The contract type '{contract}' is not a long-term commitment, which keeps the customer in a transition stage.")
        if tenure < 24:
            reasons.append(f"Tenure of {tenure} months suggests the customer is not yet fully anchored to the brand.")
        if partner == "No":
            reasons.append("The customer does not show a household stability indicator that usually supports retention.")
        return "May Become Loyal", reasons

    if churn_probability >= 0.15:
        reasons.append("The churn score remains low, but some weaker retention indicators still need attention.")
        if contract in ["One year", "Two year"]:
            reasons.append(f"The '{contract}' contract supports stronger long-term retention than month-to-month contracts.")
        if tenure >= 24:
            reasons.append(f"Tenure of {tenure} months reflects a meaningful relationship with Djezzy.")
        if total > 0:
            reasons.append(f"Total charges of {total:.2f} DZD show a sustained usage history over time.")
        return "Stable", reasons

    reasons.append("The model indicates a low probability of churn for this customer.")
    if contract in ["One year", "Two year"]:
        reasons.append(f"The '{contract}' contract is a strong retention signal for this profile.")
    if tenure >= 24:
        reasons.append(f"Tenure of {tenure} months makes this customer one of the more established accounts.")
    if partner == "Yes" or dependents == "Yes":
        reasons.append("Household stability indicators are favorable and support consistent retention.")
    if senior == 0:
        reasons.append("The profile does not show high-risk characteristics associated with churn in the current model.")
    return "Loyal", reasons


def build_state_summary(state: str, churn_probability: float, customer: pd.Series, actual_outcome: str | None = None) -> str:
    contract = customer["Contract"]
    tenure = int(customer["tenure"])
    monthly = float(customer["MonthlyCharges"])

    if actual_outcome == "Left":
        return f"This customer left. The main signs were a {contract} plan, {tenure}-month tenure, and a risk score of {churn_probability * 100:.1f}%."
    if actual_outcome == "Did not leave":
        return f"This customer stayed. The risk score is {churn_probability * 100:.1f}%, but the historical record shows the customer did not leave."

    summaries = {
        "Left": f"This customer is at high churn risk ({churn_probability * 100:.1f}%). The combination of a {contract} plan, {tenure}-month tenure, and elevated monthly spend creates a clear risk profile.",
        "May Leave": f"This customer is in a moderate risk segment ({churn_probability * 100:.1f}%). The risk is driven mainly by a {contract} agreement and limited relationship depth.",
        "May Become Loyal": f"This customer is still in a transition stage ({churn_probability * 100:.1f}%). The profile is not critical yet, but retention actions could strengthen loyalty.",
        "Stable": f"This customer appears stable ({churn_probability * 100:.1f}%). The profile has moderate retention strength and does not currently show critical churn signals.",
        "Loyal": f"This customer is highly stable ({churn_probability * 100:.1f}%). The profile shows signs of solid long-term commitment and healthy engagement."
    }
    return summaries.get(state, f"This customer is classified as {state} with a churn probability of {churn_probability * 100:.1f}%.")


def build_risk_drivers(customer: pd.Series, state: str) -> list[dict]:
    contract = customer["Contract"]
    tenure = int(customer["tenure"])
    monthly = float(customer["MonthlyCharges"])
    internet = customer["InternetService"]
    payment = customer["PaymentMethod"]

    drivers = []

    if state in ["Left", "May Leave"]:
        drivers.append({"label": "Contract", "value": contract, "detail": "Longer commitments usually reduce churn risk."})
        drivers.append({"label": "Tenure", "value": f"{tenure} months", "detail": "Shorter tenure usually means lower relationship depth."})
        if monthly > 0:
            drivers.append({"label": "Monthly charges", "value": f"{monthly:.2f} DZD", "detail": "Higher monthly spend can increase sensitivity to competitor offers."})
        if payment in ["Electronic check", "Mailed check"]:
            drivers.append({"label": "Payment method", "value": payment, "detail": "Less automated payment habits are often associated with weaker retention."})
    elif state == "May Become Loyal":
        drivers.append({"label": "Contract", "value": contract, "detail": "This is a transition profile rather than a clear churn risk."})
        drivers.append({"label": "Tenure", "value": f"{tenure} months", "detail": "The customer is not yet fully positioned as a long-term loyal account."})
        drivers.append({"label": "Service", "value": internet, "detail": "Service type can influence how stable the relationship is."})
    elif state == "Stable":
        drivers.append({"label": "Contract", "value": contract, "detail": "The customer has a comparatively stable plan structure."})
        drivers.append({"label": "Tenure", "value": f"{tenure} months", "detail": "The relationship is established enough to reduce churn risk."})
        drivers.append({"label": "Usage", "value": f"{customer['TotalCharges']:.2f} DZD", "detail": "Experienced usage suggests ongoing value from the service."})
    else:
        drivers.append({"label": "Contract", "value": contract, "detail": "This contract supports long-term retention."})
        drivers.append({"label": "Tenure", "value": f"{tenure} months", "detail": "This customer has a mature relationship with the brand."})
        drivers.append({"label": "Customer signal", "value": "Low churn risk", "detail": "The profile does not show the strongest churn-risk indicators used by the model."})

    return drivers[:3]


def build_retention_plan(customer: pd.Series, state: str) -> list[dict]:
    contract = customer["Contract"]
    tenure = int(customer["tenure"])
    monthly = float(customer["MonthlyCharges"])
    payment = customer["PaymentMethod"]
    internet = customer["InternetService"]
    services = [
        customer.get("OnlineSecurity", "No"),
        customer.get("OnlineBackup", "No"),
        customer.get("DeviceProtection", "No"),
        customer.get("TechSupport", "No"),
    ]

    actions: list[dict] = []

    if contract == "Month-to-month":
        actions.append({
            "title": "Switch to a long-term value bundle",
            "offer": "Offer a 10–15% discount on a 12-month plan or family package.",
            "why": "This customer is still on a flexible contract, which usually increases churn risk.",
        })
    if tenure <= 12:
        actions.append({
            "title": "Welcome-back loyalty campaign",
            "offer": "Propose a 3-month retention bonus or free add-on after a 6-month commitment.",
            "why": "Short tenure usually means the relationship is still fragile and needs stronger reassurance.",
        })
    if payment in ["Electronic check", "Mailed check"]:
        actions.append({
            "title": "Automatic payment promotion",
            "offer": "Offer a small monthly credit for switching to direct debit or mobile payment.",
            "why": "Customers using less automated payments are often less loyal and more price-sensitive.",
        })
    if internet == "Fiber optic":
        actions.append({
            "title": "Connectivity upgrade offer",
            "offer": "Propose a faster fiber bundle or a personalized quality guarantee package.",
            "why": "Fiber customers can become highly sensitive to competitor pricing or speed complaints.",
        })
    if "No" in services:
        actions.append({
            "title": "Add protection and support package",
            "offer": "Bundle online security, tech support, or device protection at a reduced rate.",
            "why": "Missing support services makes the customer feel the plan is less valuable than alternatives.",
        })
    if monthly >= 60:
        actions.append({
            "title": "Price sensitivity offer",
            "offer": "Introduce a data or plan reduction option, loyalty voucher, or loyalty cash-back.",
            "why": "Higher spend creates more sensitivity when a competitor offers a lower price.",
        })

    if not actions:
        actions.append({
            "title": "Retention check-in",
            "offer": "Schedule a call with the customer care team to review satisfaction and identify unmet needs.",
            "why": "The account does not show a major churn trigger, but a proactive outreach still adds protection.",
        })

    if state in ["Left", "May Leave"]:
        actions = actions[:3]
    elif state == "May Become Loyal":
        actions = actions[:2]
    else:
        actions = actions[:2]

    return actions


def prepare_customer_record(row: pd.Series) -> dict:
    return {
        "customerID": row["customerID"],
        "gender": row.get("gender", "Unknown"),
        "SeniorCitizen": int(row.get("SeniorCitizen", 0)),
        "Partner": row.get("Partner", "Unknown"),
        "Dependents": row.get("Dependents", "Unknown"),
        "tenure": int(row["tenure"]),
        "PhoneService": row.get("PhoneService", "Unknown"),
        "MultipleLines": row.get("MultipleLines", "Unknown"),
        "InternetService": row.get("InternetService", "Unknown"),
        "OnlineSecurity": row.get("OnlineSecurity", "Unknown"),
        "OnlineBackup": row.get("OnlineBackup", "Unknown"),
        "DeviceProtection": row.get("DeviceProtection", "Unknown"),
        "TechSupport": row.get("TechSupport", "Unknown"),
        "StreamingTV": row.get("StreamingTV", "Unknown"),
        "StreamingMovies": row.get("StreamingMovies", "Unknown"),
        "Contract": row.get("Contract", "Unknown"),
        "PaperlessBilling": row.get("PaperlessBilling", "Unknown"),
        "PaymentMethod": row.get("PaymentMethod", "Unknown"),
        "MonthlyCharges": float(row["MonthlyCharges"]),
        "TotalCharges": float(row["TotalCharges"]),
        "Churn": int(row["Churn"]) if "Churn" in row and pd.notna(row["Churn"]) else None,
    }


def classify_customer(customer_id: str):
    df = get_dataset()
    customer = df[df["customerID"] == customer_id]
    if customer.empty:
        return {"found": False, "message": f"Customer {customer_id} was not found in the dataset."}

    row = customer.iloc[0]
    model = get_model()
    # prepare features robustly (ignore labels that may not exist)
    feature_row = row.drop(labels=[c for c in ["customerID", "Churn", "gender"] if c in row.index], errors="ignore")
    churn_probability = float(model.predict_proba(pd.DataFrame([feature_row]))[0, 1])
    predicted_churn = int(model.predict(pd.DataFrame([feature_row]))[0])
    actual_churn = int(row["Churn"]) if "Churn" in row and pd.notna(row["Churn"]) else None
    state, reasons = assign_state(row, churn_probability, df)

    if actual_churn == 1:
        state = "Left"
    elif actual_churn == 0:
        state = "Stable" if churn_probability < 0.35 else state

    actual_outcome = "Left" if actual_churn == 1 else "Did not leave" if actual_churn == 0 else "Unknown"

    if actual_outcome == "Left":
        state = "Left"
    elif actual_outcome == "Did not leave":
        state = "Stable"

    state_summary = build_state_summary(state, churn_probability, row, actual_outcome)
    risk_drivers = build_risk_drivers(row, state)
    retention_plan = build_retention_plan(row, state)

    response = {
        "found": True,
        "customer": prepare_customer_record(row),
        "prediction": {
            "actual_outcome": actual_outcome,
            "predicted_churn": "Yes" if predicted_churn == 1 else "No",
            "churn_probability": round(churn_probability, 4),
            "probability_percent": round(churn_probability * 100, 2),
        },
        "state": state,
        "state_summary": state_summary,
        "risk_drivers": risk_drivers,
        "reasons": reasons,
        "retention_plan": retention_plan,
    }
    return response


def summary_counts():
    df = get_dataset()
    summary = []
    # Load model once to avoid heavy repeated deserialization and speed up response
    model = get_model()
    # Build a feature frame and predict probabilities in batch for performance
    to_drop = [c for c in ["customerID", "Churn", "gender"] if c in df.columns]
    features = df.drop(columns=to_drop)
    try:
        probs = model.predict_proba(features)[:, 1]
    except Exception:
        # Fallback to per-row prediction if batch predict fails for any reason
        probs = []
        for _, row in df.iterrows():
            feature_row = row.drop(labels=["customerID", "Churn", "gender"])
            probs.append(float(model.predict_proba(pd.DataFrame([feature_row]))[0, 1]))

    for (_, row), prob in zip(df.iterrows(), probs):
        state, _ = assign_state(row, float(prob), df)
        summary.append({"customerID": row["customerID"], "state": state})
    counts = pd.Series([item["state"] for item in summary]).value_counts().to_dict()
    return {
        "Left": counts.get("Left", 0),
        "May Leave": counts.get("May Leave", 0),
        "Stable": counts.get("Stable", 0),
        "May Become Loyal": counts.get("May Become Loyal", 0),
        "Loyal": counts.get("Loyal", 0),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/customer/<customer_id>")
def customer_lookup(customer_id: str):
    try:
        return jsonify(classify_customer(customer_id))
    except Exception as exc:  # pragma: no cover
        response = {"found": False, "message": f"Error while searching: {exc}"}
        return jsonify(response), 500


@app.route("/api/summary")
def summary_api():
    return jsonify(summary_counts())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
