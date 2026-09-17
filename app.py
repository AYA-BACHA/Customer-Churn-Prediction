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
            reasons.append("Month-to-month contract is associated with a high churn risk.")
        if tenure <= 12:
            reasons.append(f"Tenure is only {tenure} months, which is a short relationship period.")
        if monthly >= median_monthly:
            reasons.append(f"MonthlyCharges of {monthly:.2f} DZD are above the portfolio median.")
        if payment in ["Electronic check", "Mailed check"]:
            reasons.append(f"Payment method '{payment}' is associated with weaker retention in this customer profile.")
        if not reasons:
            reasons.append("The model assigns a very high churn probability for this customer profile.")
        return "Left", reasons

    if churn_probability >= 0.5:
        if contract == "Month-to-month":
            reasons.append("Month-to-month contract is a key risk signal in this profile.")
        if tenure <= 18:
            reasons.append(f"Tenure is {tenure} months, which is shorter than a long-term retention pattern.")
        if internet == "Fiber optic":
            reasons.append("Fiber optic service is associated with higher churn sensitivity in this dataset pattern.")
        if payment in ["Electronic check", "Mailed check"]:
            reasons.append(f"Payment method '{payment}' is a weaker retention signal than automatic billing.")
        if not reasons:
            reasons.append("The model is moderately concerned about churn for this customer.")
        return "May Leave", reasons

    if churn_probability >= 0.35:
        reasons.append("The churn score is in a caution zone but not yet a strong churn signal.")
        if contract != "Two year":
            reasons.append(f"Contract type '{contract}' is not a long-term commitment pattern.")
        if tenure < 24:
            reasons.append(f"Tenure of {tenure} months suggests the customer is still in a transitional stage.")
        if partner == "No":
            reasons.append("Customer is not currently partnered, which can be associated with weaker retention patterns.")
        return "May Become Loyal", reasons

    if churn_probability >= 0.15:
        reasons.append("The model shows a low but not negligible churn risk.")
        if contract in ["One year", "Two year"]:
            reasons.append(f"Contract type '{contract}' supports a more stable retention pattern.")
        if tenure >= 24:
            reasons.append(f"Tenure of {tenure} months reflects an established customer relationship.")
        if total > 0:
            reasons.append(f"TotalCharges of {total:.2f} DZD indicate sustained usage over time.")
        return "Stable", reasons

    reasons.append("The model indicates a low churn probability.")
    if contract in ["One year", "Two year"]:
        reasons.append(f"A '{contract}' contract reflects a stronger long-term retention pattern.")
    if tenure >= 24:
        reasons.append(f"Tenure of {tenure} months reflects a mature customer relationship.")
    if partner == "Yes" or dependents == "Yes":
        reasons.append("Household stability indicators are favorable for retention.")
    if senior == 0:
        reasons.append("This profile does not show the strongest churn-risk indicators used in the model.")
    return "Loyal", reasons


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
    feature_row = row.drop(labels=["customerID", "Churn", "gender"])
    churn_probability = float(model.predict_proba(pd.DataFrame([feature_row]))[0, 1])
    predicted_churn = int(model.predict(pd.DataFrame([feature_row]))[0])
    state, reasons = assign_state(row, churn_probability, df)

    response = {
        "found": True,
        "customer": prepare_customer_record(row),
        "prediction": {
            "predicted_churn": "Yes" if predicted_churn == 1 else "No",
            "churn_probability": round(churn_probability, 4),
            "probability_percent": round(churn_probability * 100, 2),
        },
        "state": state,
        "reasons": reasons,
    }
    return response


def summary_counts():
    df = get_dataset()
    summary = []
    for customer_id, row in df.iterrows():
        model = get_model()
        feature_row = row.drop(labels=["customerID", "Churn", "gender"])
        prob = float(model.predict_proba(pd.DataFrame([feature_row]))[0, 1])
        state, _ = assign_state(row, prob, df)
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
