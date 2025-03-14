from app import model_pred
import pytest

new_data = {'customer_id': 2243629,
            'credit_lines_outstanding': 5,
            'loan_amt_outstanding': 5181,
            'total_debt_outstanding': 23957,
            'income': 82865,
            'years_employed': 2,
            'fico_score': 559,
            }


def test_predict():
    prediction = model_pred(new_data)
    print(f"Data: {new_data}, Prediction: {prediction}")
    assert prediction == 1, "incorrect prediction" 
