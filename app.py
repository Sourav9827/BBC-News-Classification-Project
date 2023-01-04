from flask import Flask, request, app, render_template, Response
from src.utils import *

import warnings
warnings.filterwarnings("ignore")


app=Flask(__name__)

app.config['SECRET_KEY'] = 'mysecretkey'
 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_api_cat')
def predict_api_cat():
    return render_template('prediction.html')

@app.route('/predict_cat', methods=['GET','POST'])
def predict_cat():

    output = category_predict(request.form['text_file'])
    return render_template('prediction.html', prediction_text="The news is a {} news".format(output))

@app.route('/predict_api_df', methods=['GET','POST'])
def predict_api_df():
    return render_template('predictcsv.html')

@app.route('/predict_df', methods=['GET','POST'])
def predict_df():
    df = df_predict(request.files['df_file'])
    return Response(df.to_html())

if __name__=="__main__":
    app.run(debug=True)