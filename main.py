from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
nltk.download('wordnet')
from gensim.models import Word2Vec
nltk.download('stopwords')
import string

app = FastAPI()
w2v_model = Word2Vec.load("w2v_model.model") 
loaded_model = tf.keras.models.load_model("bilstm_word2vec_model.h5")

templates = Jinja2Templates(directory="templates")
stopwords=set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

max_length = 2000
dim = 300
vocab_size = 160_00
output_dim=300


def cleanWord(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[^a-zA-Z]',' ',text)
    text = text.split()
    text = ' '.join(text)
    return text
def predict(news: str):
    processed_data=cleanWord(news)
    tokenized_text = processed_data.split()
    word_vectors = [w2v_model.wv.get_vector(word) for word in tokenized_text if word in w2v_model.wv.key_to_index]
    max_length = 2000
    valid_word_vectors = []
    for seq in word_vectors:
        valid_indices = [idx for idx in seq if 0 <= idx < 16000]
        valid_word_vectors.append(valid_indices)
    word_padded= pad_sequences(valid_word_vectors,padding='post', maxlen=max_length)
    prediction = loaded_model.predict(word_padded)
    final_prediction = np.mean(prediction)
    threshold = 0.5  
    binary_predictions = (final_prediction > threshold).astype(int)
    if binary_predictions == 1:
      return ("IT IS A REAL NEWS")
    else:
      return ("IT IS A FAKE NEWS")
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def get_data(request: Request, news: str = Form(...)):
    prediction = predict(news)
    return templates.TemplateResponse("index.html", {"request": request, "prediction": str(prediction)})
