import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
model = None 

def train_model():
    data = pd.read_csv("artist.csv")

    data.drop(["school","age"], axis=1, inplace=True)
    data_dum = data.copy()
    categorical_d = {'yes': 1, 'no': 0}
    data_dum['Astatus'] = data_dum['Astatus'].map(categorical_d)
    data_dum['paid'] = data_dum['paid'].map(categorical_d)
    data_dum['activities'] = data_dum['activities'].map(categorical_d)
    categorical_d = {'F': 1, 'M': 0}
    data_dum['sex'] = data_dum['sex'].map(categorical_d)
    categorical_d = {'U': 1, 'R': 0}
    data_dum['address'] = data_dum['address'].map(categorical_d)
    categorical_d = {'I': 1, 'B': 0}
    data_dum['status'] = data_dum['status'].map(categorical_d)
    categorical_d = {'acrylic': 1, 'watercolor': 0}
    data_dum['mode'] = data_dum['mode'].map(categorical_d)
    categorical_d = {'mother': 0, 'father': 1, 'other': 2}
    data_dum['guardian'] = data_dum['guardian'].map(categorical_d)

    # Split data into features and target
    x = data_dum.drop("Final", axis=1)
    y = data_dum['Final']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=44)

    global model
    model = LinearRegression()
    model.fit(X_train, y_train)

train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Predicted Final Score: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
