
# Import necessary libraries
import logging
import joblib                                   # to load the serialized model
import pandas as pd                             # for data handling
from flask import Flask, request, jsonify       # to build the REST API

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("superkart")

# Initialize the Flask application
superkart_api = Flask("SuperKart")

# Load the trained model pipeline (preprocessing + model) once at start-up
model = joblib.load("superkart_model.joblib")

# Input contract: the same columns, in the same order, that the model was trained on
NUMERIC_FEATURES = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age_Years"]
CATEGORICAL_FEATURES = ["Product_Sugar_Content", "Store_Size", "Store_Location_City_Type",
                        "Store_Type", "Product_Id_char", "Product_Type_Category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Categories seen during training (used to warn about values the model has never seen)
_encoder = model.named_steps["preprocessor"].named_transformers_["cat"]
KNOWN_CATEGORIES = {col: set(cats) for col, cats in zip(CATEGORICAL_FEATURES, _encoder.categories_)}


def clean_frame(df):
    """Validate a DataFrame of inputs. Returns (clean_df, error_message or None)."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return None, f"Missing required column(s): {missing}"
    df = df[FEATURES].copy()                    # extra columns are ignored, order is fixed
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = df.index[df[NUMERIC_FEATURES].isnull().any(axis=1) | df[CATEGORICAL_FEATURES].isnull().any(axis=1)]
    if len(bad) > 0:
        return None, f"Missing or non-numeric values in row(s): {[int(i) for i in bad[:10]]}"
    if (df[NUMERIC_FEATURES] < 0).any().any():
        return None, "Numeric features must not be negative."
    return df, None


def unseen_categories(df):
    """List category values that were not present in the training data."""
    notes = []
    for col, known in KNOWN_CATEGORIES.items():
        unseen = sorted(set(df[col].astype(str)) - known)
        if unseen:
            notes.append(f"{col}: unseen value(s) {unseen} were not in the training data")
    return notes


# Home page
@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart System"


# Health check
@superkart_api.get("/health")
def health():
    return jsonify({"status": "ok"})


# Endpoint: predict total sales for ONE product in ONE store (JSON in, JSON out)
@superkart_api.post("/v1/predict")
def predict_sales():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    df, error = clean_frame(pd.DataFrame([payload]))
    if error:
        return jsonify({"error": error}), 400

    prediction = float(model.predict(df)[0])
    response = {"Sales": round(prediction, 2)}
    notes = unseen_categories(df)
    if notes:
        response["warnings"] = notes
    return jsonify(response)


# Endpoint: predict total sales for MANY rows (CSV file upload in field 'file')
@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "Upload a CSV file in the form field named 'file'."}), 400
    try:
        raw = pd.read_csv(file)
    except Exception:
        return jsonify({"error": "The uploaded file could not be read as a CSV."}), 400
    if raw.empty:
        return jsonify({"error": "The uploaded CSV has no rows."}), 400

    df, error = clean_frame(raw)
    if error:
        return jsonify({"error": error}), 400

    predictions = model.predict(df).tolist()
    # Row index (as in the uploaded file) -> predicted sales
    return jsonify({str(i): round(float(p), 2) for i, p in enumerate(predictions)})


@superkart_api.errorhandler(Exception)
def unexpected_error(exc):
    from werkzeug.exceptions import HTTPException
    if isinstance(exc, HTTPException):
        return jsonify({"error": exc.description}), exc.code
    logger.exception("Unexpected error")
    return jsonify({"error": "Internal server error while predicting."}), 500


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860)
