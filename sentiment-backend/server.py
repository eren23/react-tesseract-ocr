from flask import Flask, request
from threading import Thread
from flask_cors import CORS

from sentiment_classifier import classifier
from sentiment_classifier import remove_noise
from nltk.tokenize import word_tokenize


app = Flask('')
CORS(app)
# app.debug=True

@app.route('/sentiment/sentence', methods=["GET", "POST"])
def home():
    if request.method == 'POST':
      sentence_from_req = request.json["sentence"]
      print(sentence_from_req)
    sentence = "this model sucksss"
    custom_tokens = remove_noise(word_tokenize(sentence_from_req))
    is_negative = classifier.classify(dict([token, True] for token in custom_tokens))
    return is_negative

def run():
  app.run(host='0.0.0.0',port=8080)

def keep_alive():
    t = Thread(target=run)
    t.start()

keep_alive()