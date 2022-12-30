from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from transformers import AutoTokenizer,TFDistilBertModel, DistilBertConfig
from transformers import TFAutoModel

import warnings
warnings.filterwarnings("ignore")


app=Flask(__name__)

app.config['SECRET_KEY'] = 'mysecretkey'

model = tf.keras.models.load_model('model/model.h5', custom_objects={'TFDistilBertModel': TFDistilBertModel})

# Creating tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenization of the data
def text_encode(text, tokenizer, max_len=100):
    tokens = text.apply(lambda x: tokenizer(x,return_tensors='tf', 
                                            truncation=True,
                                            padding='max_length',
                                            max_length=max_len, 
                                            add_special_tokens=True))
    input_ids= []
    attention_mask=[]
    for item in tokens:
        input_ids.append(item['input_ids'])
        attention_mask.append(item['attention_mask'])
    input_ids, attention_mask=np.squeeze(input_ids), np.squeeze(attention_mask)

    return [input_ids,attention_mask]

def category_predict(text):
  text_df = pd.DataFrame(data=[text], columns=['Text'])
  df_text_input_ids, df_text_attention_mask = text_encode(text_df['Text'], tokenizer, max_len=100)
  df_text_predict = model.predict([df_text_input_ids, df_text_attention_mask])
  
  text_df.reset_index(inplace = True)
  column_values = ['business', 'entertainment', 'politics', 'sport', 'tech']
  df_category_matrix = pd.DataFrame(data = df_text_predict, columns = column_values)

  df_test_result = pd.concat([text_df, df_category_matrix], axis=1)
  category = pd.DataFrame(df_test_result.set_index('Text').idxmax(axis = 'columns'), columns = ['Category']).iloc[:1]['Category'][0]
  return category

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_api',methods=['GET','POST'])
def predict_api():

    output = category_predict(request.form['text_file'])
    return render_template('prediction.html', prediction_text="The news is a {} news".format(output))


if __name__=="__main__":
    app.run(debug=True)