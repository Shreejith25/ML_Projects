import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

data = pd.read_excel("iris .xls")

## Splitting data
x = data.drop('Classification',axis = 1)
y = data['Classification']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.3,random_state=0)

from sklearn.linear_model import LogisticRegression ## Importing logistic Regresion
lo_re = LogisticRegression()
lo_re=lo_re.fit(x_train,y_train)#Fitting the model
pickle.dump(lo_re,open('model.pkl','wb') )