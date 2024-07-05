import array
from flask import Flask,request,render_template
import pickle
import pandas as pd
import numpy as np

app  = Flask(__name__)


def transform_decode(val):
    y_vals = []
    for i in val:
        if i == 0:
            y_vals.append("love🥰")
        elif i == 1:
            y_vals.append("anger😡")
        elif i == 2:
            y_vals.append('surprise 😮')
        elif i == 3:
            y_vals.append('happiness 😃')
        elif i ==4:
            y_vals.append( 'sadness 😔')
        elif i == 5:
            y_vals.append("fear 😨")
    return y_vals

vectoriser = pickle.load(open("vectoriser.pkl","rb"))

file = open("text_emotion.pkl","rb")
model = pickle.load(file)

@app.route("/")

def home():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    input_feature = [request.form.get("Text")]
    print(input_feature)
    Vectorize = vectoriser.transform(input_feature)
    Value_pridected = transform_decode(model.predict(Vectorize))
    Value_pridected = Value_pridected[0]
    print(Value_pridected)
    return render_template("index.html", jk = Value_pridected)

if __name__ == "__main__":
    app.run(debug=True)


