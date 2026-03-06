from flask import Flask, request, render_template, jsonify
import sys

from src.pipeline.prediction_pipeline import PredictPipeline, CustomData
from src.logger import logger
from src.exception import CustomException

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = CustomData(
            age=float(request.form.get("age")),
            gender=request.form.get("gender"),
            income=float(request.form.get("income")),
            income_stability=request.form.get("income_stability"),
            profession=request.form.get("profession"),
            employment_type=request.form.get("employment_type"),
            location=request.form.get("location"),
            loan_amount_request=float(request.form.get("loan_amount_request")),
            current_loan_expenses=float(request.form.get("current_loan_expenses", 0)),
            credit_card_status=request.form.get("credit_card_status"),
            property_location=request.form.get("property_location"),
            credit_score=float(request.form.get("credit_score")),
            no_of_defaults=int(request.form.get("no_of_defaults", 0)),
            dependents=int(request.form.get("dependents", 0)),
            property_price=float(request.form.get("property_price")),
        )

        df = data.get_data_as_dataframe()
        pipeline = PredictPipeline()
        result = pipeline.predict(df)

        logger.info(f"Prediction result: {result}")
        return render_template("index.html", result=result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        error_result = {"error": str(e)}
        return render_template("index.html", result=None, error=str(e)), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Loan Sanction Prediction API"}), 200


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
