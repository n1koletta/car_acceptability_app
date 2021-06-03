from flask import Flask, request, Response, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#load data
df = pd.read_csv("./dataset/car.csv")

#extract categoric features
X = df.iloc[:,:-1].values

#extract outcome variable
y = df.iloc[:, -1].values

#onehotencoding
ohe = OneHotEncoder()
categoric_data = ohe.fit_transform(X).toarray()
categoric_df = pd.DataFrame(categoric_data)
categoric_df.columns = ohe.get_feature_names()

X_final = categoric_df

#train model
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_final, y)

#create flask instance
app = Flask(__name__)

#create api
@app.route('/api', methods=['GET', 'POST'])
def predict():
    #get data from request
    data = request.get_json(force=True)
    requestData = np.array([data["buying"], data["maint"], data["doors"], data["persons"], data["lug_boot"], data["safety"]])
    requestData = np.reshape(requestData, (1, -1))
    requestData = ohe.transform(requestData).toarray()

    #data_final = np.column_stack((data_categoric))
    data_final = pd.DataFrame(requestData, dtype=object)

    #make predicon using model
    prediction = rfc.predict(data_final)
    return Response(json.dumps(prediction[0]))
