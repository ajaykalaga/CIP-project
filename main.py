
from unicodedata import numeric
import streamlit as st
from sklearn import datasets
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


df=pd.read_csv("GlobalTemperatures.csv")
df = df.dropna()
df['dt'] = pd.to_datetime(df['dt'], 
format = '%Y-%m-%d',
errors = 'coerce')

df['dt'] = df['dt'].dt.month

X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

# 


# website implementation
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;"> Weather Forecasting using ML</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    # month = st.text_input("Enter the month")
    # LandAverageTemperature = st.text_input("LandAverageTemperature")
    # LandAverageTemperatureUncertainty = st.text_input("LandAverageTemperatureUncertainty")
    # LandMaxTemperature = st.text_input("LandMaxTemperature")
    # LandMaxTemperatureUncertainty = st.text_input("LandMaxTemperatureUncertainty")
    # LandMinTemperature = st.text_input("LandMinTemperature")
    # LandMinTemperatureUncertainty = st.text_input("LandMinTemperatureUncertainty")
    # LandAndOceanAverageTemperature = st.text_input("LandAndOceanAverageTemperature") 
    a = st.selectbox('Month',("1","2","3","4","5","6","7","8","9","10","11","12"))
    b = st.number_input("LandAverageTemperature")
  
    c = st.number_input("LandAverageTemperatureUncertainty") 
    d = st.number_input("LandMaxTemperature")
    e = st.number_input("LandMaxTemperatureUncertainty")
    f = st.number_input("LandMinTemperature")
    g = st.number_input("LandMinTemperatureUncertainty")
    h = st.number_input("LandAndOceanAverageTemperature")
    
    result =""
    regressor.predict([[a,b,c,d,e,f,g,h]])

    if st.button("Predict"):

        result = regressor.predict([[a,b,c,d,e,f,g,h]])
        result = str(result)
        new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 42px;">'+ result +'</p>'
        st.markdown(new_title, unsafe_allow_html=True)
         
        st.write('The predicted Land And Ocean Average Temperature Uncertainty is   ', result)
     

if __name__=='__main__': 
    main()
