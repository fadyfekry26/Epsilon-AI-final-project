from flask import Flask,render_template,request
import joblib
from helpers.dummies import *
import numpy as np
app=Flask(__name__)

model=joblib.load('models/model1')
scaler=joblib.load('models/scale1')

@app.route('/',methods=['GET'])
def index():
    return render_template('index (1).html')


## post method added to the index html instead of get 
## post method added and some modifications added to the function to take the data correctly from the web page
@app.route('/predict',methods=['GET', 'POST'])
def predict():
    # all_data=request.args
    if request.method == 'POST' :
        age=float(request.form['age'])
        job=float(request.form['job'])
        marital=float(request.form['marital'])
        education=float(request.form['education'])
        default=float(request.form['Default'])
        housing=float(request.form['Housing'])
        loan=float(request.form['Loan'])
        contact=float(request.form['Contact_dummies'])
        month=float(request.form['month'])
        day_of_week=float(request.form['day_of_week'])
        poutcome=float(request.form['poutcome'])
        duration=float(request.form['duration'])
        previous=float(request.form['previous'])
        campaign=float(request.form['campaign'])
    data=[age,job,marital,education,default,housing,loan,contact,month,day_of_week,campaign,poutcome,duration,previous]
    data_scaled=scaler.transform([data])
    pred=model.predict(data_scaled)[0]
    return render_template('prediction.html',profit=pred)
    #return str(data)


if __name__=='__main__':
    app.run()