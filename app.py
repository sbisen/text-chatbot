from flask import Flask, render_template, request

import pandas as pd
import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25L
from scipy import sparse


df_csv= pd.read_csv('aggregated-hw3-ratings.train.csv', header=None)
df_csv.columns= ['message_id','response_id','score']
df_csv.drop(columns='message_id', inplace=True)
df_tsv = pd.read_table('chatbot-replies.tsv.gz', sep='\n\t|\t', engine='python',compression='gzip')
combine = df_tsv.merge(df_csv, how='inner', on=['response_id'])
get_reply=[]
for i in combine.response :
    get_reply.append(i)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def get_bot_response():
    what_the_user_said = request.args.get('msg')


    all_token= [t.split(" ") for t in get_reply]
    bm25l = BM25L(all_token)
    tokens= what_the_user_said.split(" ")
    chatbot_reply = bm25l.get_top_n(tokens, get_reply, n=1)
    return str(chatbot_reply[0])

if __name__ == "__main__":

    
    # IMPLEMENTATION HINT: you probably want to load and cache your conversation
    # database (provided by us) here before the chatbot runs
       
    app.run()
