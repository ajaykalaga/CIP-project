from multiapp import MultiApp
from apps import home, model # import your app modules here
from unicodedata import numeric
from calendar import month
import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

df= pd.read_csv("GlobalTemperatures.csv")
df = df.dropna()
df['dt'] = pd.to_datetime(df['dt'], 
format = '%Y-%m-%d',
errors = 'coerce')

df['dt'] = df['dt'].dt.month



X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float32)

X = np.nan_to_num(X) 

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



st.write("""
# Wants to predict the temperature? Here you go!!

""")

var = st.sidebar.selectbox(
    'Select Month',
    ('January', 'February', 'March','April','May','June','July','August','September','October','November','December',)
)
a=0
if var=='January':
    a=1
elif var=='February':
    a==2
elif var=='March':
    a==3
elif var=='April':
    a==4
elif var=='May':
    a==5
elif var=='June':
    a==6
elif var=='July':
    a==7
elif var=='August':
    a==8
elif var=='September':
    a==9
elif var=='October':
    a==10
elif var=='November':
    a==11
elif var=='December':
    a==12
    
    
b = st.sidebar.slider(("LandMaxTemperature"),min_value=0.0, max_value=100.0)
    
c = st.sidebar.slider(("LandMinTemperature"),min_value=0.0, max_value=100.0)
    
d= st.sidebar.slider(("LandAndOceanAverageTemperature"),min_value=0.0, max_value=100.0)

def main():       
    # front end elements of the web page 
    html_temp = """ """
    result =""
    regressor.predict([[a,b,c,d]])

    if st.button("Predict"):

        result = regressor.predict([[a,b,c,d]])
        result = str(result)
        new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 42px;">'+ result +'</p>'
        st.markdown(new_title, unsafe_allow_html=True)
         
        st.write('The predicted Land  Average Temperature  is   ', result)
    # plt.plot(y_pred, color="red")
    # plt.xlabel("y_predict")
    # plt.ylabel("y_test")
    # plt.plot(y_test,color="grey")
    # plt.show()
#     pca = PCA(2)
#     X_projected = pca.fit_transform(X)

#     x1 = X_projected[:, 0]
#     x2 = X_projected[:, 1]

#     fig = plt.figure()

#     plt.scatter(x1, x2,
#         c=y, alpha=0.8,
#         cmap='viridis')

#     plt.xlabel('y_predict')
#     plt.ylabel('y_test')
#     plt.colorbar()

# #plt.show()
#     st.pyplot(fig) 

if __name__=='__main__': 
    main()



app = MultiApp()

# st.markdown("""
# # Multi-Page App
# This multi-page app is using the [streamlit-multiapps](https://github.com/upraneelnihar/streamlit-multiapps) framework developed by [Praneel Nihar](https://medium.com/@u.praneel.nihar). Also check out his [Medium article](https://medium.com/@u.praneel.nihar/building-multi-page-web-app-using-streamlit-7a40d55fa5b4).
# """)

# Add all your application here
app.add_app("Land max and min temperature", home.app)

app.add_app("Land and ocean temperature", model.app)
# The main app
app.run()