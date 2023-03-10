import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import TFDistilBertModel, AutoTokenizer
import requests
import os

# Creating tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def load_model(url):
    files = os.listdir('model')
    if 'model.h5' not in files:
        response = requests.get(url)
        with open('model/model.h5', 'wb') as f:
            f.write(response.content)
    model = tf.keras.models.load_model('model/model.h5', custom_objects={'TFDistilBertModel': TFDistilBertModel})
    return model

url = 'https://bbc-news-classification-model.s3.ap-south-1.amazonaws.com/model.h5'
model = load_model(url)

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

def df_predict(file):
    df = pd.read_csv(file, header=0, encoding='cp1252')
    df_input_ids, df_attention_mask = text_encode(df['Text'], tokenizer, max_len=100)
    df_predict = model.predict([df_input_ids, df_attention_mask])
    column_values = ['business', 'entertainment', 'politics', 'sport', 'tech']
    df2 = pd.DataFrame(data = df_predict, columns = column_values)  
    df_test_result = pd.concat([df, df2], axis=1)
    df_result = pd.DataFrame(df_test_result.set_index('Text').idxmax(axis = 'columns'), columns = ['Category'])
    return df_result

    
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

