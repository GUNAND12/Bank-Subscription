import pandas as pd
import numpy as np

from flask import Flask, request,render_template

import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
from sklearn import *
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)
#import warnings


test_inp = pickle.load(open("test_inpv1", "rb"))
test_out = pickle.load(open("test_outv1", "rb"))
feat_num = pickle.load(open("feat_numv1", "rb"))
feat_cat = pickle.load(open("feat_catv1", "rb"))
data_pipeline = pickle.load(open("data_pipelinev1", "rb"))
lr = pickle.load(open("lr_model", "rb"))
gb = pickle.load(open("gb_model", "rb"))
#xg_model = pickle.load(open("final_xg_model2", "rb"))
#xg_model = pickle.load(open("pima.pickle.dat", "rb"))
#import xgboost as xgb
#bst = xgb.Booster({'nthread': 4})  # init model
#model=xgb.XGBClassifier()
#bank_model=model.load_model('xgmodel.model')





@app.route('/',methods=["GET"])
def Home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    """Predict the Loan Status
        Enter the following customer details.
        ---
        parameters:
          - name: age
            in: query
            type: number
            required: true
          - name: job_type
            in: query
            type: string
            required: true
          - name: marital
            in: query
            type: string
            required: true
          - name: education
            in: query
            type: string
            required: true
          - name: default
            in: query
            type: string
            required: true
          - name: balance
            in: query
            type: number
            required: true
          - name: housing_loan
            in: query
            type: string
            required: true
          - name: personal_loan
            in: query
            type: string
            required: true
          - name: contact
            in: query
            type: string
            required: true
          - name: day
            in: query
            type: number
            required: true
          - name: month
            in: query
            type: string
            required: true
          - name: duration
            in: query
            type: number
            required: true
          - name: campaign
            in: query
            type: number
            required: true
          - name: pdays
            in: query
            type: string
            required: true
          - name: previous
            in: query
            type: number
            required: true
          - name: poutcome
            in: query
            type: string
            required: true
        responses:
            200:
                description: The output values

        """
    age = request.form["age"]
    job_type = request.form["job_type"]
    marital = request.form["marital"]
    education = request.form["education"]
    default = request.form["default"]
    balance = request.form["balance"]
    housing_loan = request.form["housing_loan"]
    personal_loan = request.form["personal_loan"]
    contact = request.form["contact"]
    day = request.form["day"]
    month = request.form["month"]
    duration = request.form["duration"]
    campaign = request.form["campaign"]
    pdays = request.form["pdays"]
    previous = request.form["previous"]
    poutcome = request.form["poutcome"]

    test = pd.DataFrame([[age, job_type, marital, education, default, balance,
                          housing_loan, personal_loan, contact, day, month, duration,
                          campaign, pdays, previous, poutcome]])
    test.columns = test_inp.columns
    xtest1 = pd.concat([test_inp, test])

    xtest1_clean = data_pipeline.fit_transform(xtest1)
    ypred = lr.predict(xtest1_clean)
    if ypred[-1]==0:

        return render_template('result.html',prediction_text="The client will NOT subscribe term deposit")
    else:
        return render_template('result.html',prediction_text="The client will  subscribe the term deposit")
    #return "The Bank Status of the customer is :" + str(ypred[-1])







if __name__=="__main__":
    app.run(debug=True)
