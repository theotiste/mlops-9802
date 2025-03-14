from flask import Flask, render_template, request
import pickle
import pandas as pd
import os
from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments
from dotenv import load_dotenv   
import datetime

# Charger les variables d'environnement
load_dotenv()
ARIZE_SPACE_KEY = os.getenv("SPACE_KEY")
ARIZE_API_KEY = os.getenv("API_KEY")

if not ARIZE_SPACE_KEY or not ARIZE_API_KEY:
    raise ValueError(" ERREUR : Les clés API Arize ne sont pas définies.")

# Initialiser Arize Client
arize_client = Client(space_key=ARIZE_SPACE_KEY, api_key=ARIZE_API_KEY)

# Définition du schéma
schema = Schema(
    prediction_id_column_name="prediction_id",
    timestamp_column_name="timestamp",
    feature_column_names=["Customer_id", "Credit_line_outstanding", "Loan_amt_outstanding",
                          "Total_debt_outstanding", "Income", "Years_employed", "Fico_score"],
    prediction_label_column_name="prediction_label",
    actual_label_column_name="actual_label"
)

app = Flask(__name__)
model = pickle.load(open("catboost_model-1.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupération des données utilisateur
        features = {key: float(request.form[key]) for key in request.form}
        prediction = model.predict(pd.DataFrame([features]))[0]

        # Envoi des données à Arize
        timestamp = pd.Timestamp.now()
        data = {**features, "prediction_id": str(timestamp.timestamp()), "timestamp": timestamp, "prediction_label": int(prediction)}
        dataframe = pd.DataFrame([data])
        arize_client.log(dataframe=dataframe, model_id="theo_model", model_version="v1", model_type=ModelTypes.SCORE_CATEGORICAL, environment=Environments.PRODUCTION, schema=schema)

        return render_template("index.html", prediction_text="Loan granted" if prediction == 0 else "Make an appointment with your banker")

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
