import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

data = pd.read_excel("iris .xls")

# Missing Value Handled
data['SL'].fillna(round(data['SL'].median(),1),inplace=True)
data['SW'].fillna(round(data['SW'].median(),1),inplace=True)
data['PL'].fillna(round(data['PL'].median(),1),inplace=True)

# Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Classification']= le.fit_transform(data['Classification'])

## Splitting data
x = data.drop('Classification',axis = 1)
y = data['Classification']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.3,random_state=0)

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf.fit(x_train,y_train)#Fitting the model
pickle.dump(lo_re,open('model.pkl','wb') )