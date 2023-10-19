from flask import Flask,render_template,request,jsonify
import pickle
import pandas as pd

app = Flask(__name__)

knn = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['post'])
def predict():
    age=request.form.get('age')
    salary=request.form.get('salary')

    result = knn.predict([[int(age),int(salary)]])[0]

    if result==1:
        return jsonify({'label':1})
    else:
        return jsonify({'label':-1})

if __name__=='__main__':
    app.run(debug=True)