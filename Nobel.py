import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier



st.write(''' # Nobel Prize category prediction''')
st.image("Nobel.png", caption="Its creator was the Swedish inventor Alfred Nobel through his will in 1895.")

st.header('Motivation')

def user_input_features():
  # Entrada
  texto = st.text_input("Enter the text to be evaluated", value="")


  user_input_data = {'Motivation': texto}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

url = "https://raw.githubusercontent.com/pamelaedi153-cmyk/ExamenFinal-ModuloIV/refs/heads/main/nobel_final.csv"

nobel =  pd.read_csv(url, encoding='utf-8')

#codificación numérica manual (Label Encoding)
nobel['label_num'] = nobel['Category'].map({'chemistry': 0,'economics': 1,'literature': 2,'medicine': 3,'peace': 4,'physics': 5})

X = nobel.Motivation
y = nobel.label_num



vect = CountVectorizer()

X_dtm = vect.fit_transform(X)

#nb = MultinomialNB()
#nb.fit(X_dtm, y)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_dtm, y)
prediction = ""

df_dtm = vect.transform(df['Motivation'])
prediction = rf.predict(df_dtm)

#{'physics':0, 'medicine':1, 'peace':2, 'literature':3, 'chemistry':4, 'economics':5}
#'Physics', 'Medicine', 'Peace', 'Literature', 'Chemistry', 'Economics'
st.subheader('Predicción')
if prediction == 0:
  st.write('Physics')
elif prediction == 1:
  st.write('Medicine')
elif prediction == 2:
  st.write('Peace')
elif prediction == 3:
  st.write('Literature')
elif prediction == 4:
  st.write('Chemistry')
elif prediction == 5:
  st.write('Economics')
else:
  st.write('Sin predicción')
