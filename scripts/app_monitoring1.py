from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("catboost_model-1.pkl", "rb"))
                                        

def model_pred(features):
    test_data = pd.DataFrame([features])
    prediction = model.predict(test_data)
    return int(prediction[0])

@app.route("/", methods=["GET"])
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        Customer_id = int(request.form["Customer_id"])
        Credit_line_outstanding = int(request.form["Credit_line_outstanding"])
        Loan_amt_outstanding = float(request.form["Loan_amt_outstanding"])
        Total_debt_outstanding = float(request.form["Total_debt_outstanding"])
        Income = float(request.form["Income"])
        Years_employed = int(request.form["Years_employed"])
        Fico_score = int(request.form["Fico_score"])
        
        prediction = model.predict(
            [[Customer_id, Credit_line_outstanding, Loan_amt_outstanding, Total_debt_outstanding, Income, Years_employed, Fico_score]]
        )
        
        if prediction[0] == 1:
            return render_template(
                "index.html",
                prediction_text="Make an appointment with your banker",
            )
        else:
            return render_template(
                "index.html",
                prediction_text="loan granted"
            )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)      
