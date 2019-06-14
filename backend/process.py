import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from flask import Flask, request, jsonify
import pickle
import json

app = Flask(__name__)
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('model.pkl','rb'))

@app.route('/', methods=['GET'])
def hello():
    return "hello"

@app.route('/overbook',methods=['POST'])
def overbook():
    data = request.get_json(force=True)

    data = pd.DataFrame(data, orient='columns')

    data = data.drop(['IsCanceled', 'Company', 'ReservationStatus'], axis = 1)
    
    categorical = ['ReservationStatusDate', 'Agent', 'ArrivalDateMonth', 'AssignedRoomType', 'Country', 'CustomerType', 'DepositType', 'DistributionChannel', 'IsRepeatedGuest', 'MarketSegment', 'Meal', 'ReservedRoomType']
    
    for cat in categorical :
        data[cat] = pd.factorize(data[cat].values)

    data = scaler.transform(data)

    prediction = model.predict([data])

    print(jsonify(prediction))
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=True)   