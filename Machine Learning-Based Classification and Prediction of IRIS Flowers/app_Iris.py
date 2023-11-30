from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('/index.html')


@app.route('/predict',methods=['POST'])
def predict():
   
    petal_length = request.form['pl']
    sepal_length = request.form['sl']
    petal_width = request.form['pw']
    sepal_width = request.form['sw']
    sample_data = [sepal_length,sepal_width,petal_length,petal_width]
    clean_data = [float(i) for i in sample_data]
    feature = np.array(clean_data).reshape(-1,4)
   
    output= model.predict(feature)
   
    output=output.item()  
    
    return render_template ('result.html',prediction_text=" The Iris Flower is {}".format(output))
if __name__=='__main__':
    app.run(port=8000)