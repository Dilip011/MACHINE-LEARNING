#SET 1 QUESTION 2


import pandas as pd
import numpy as np

data = [
        ['Sunny','Warm','Normal','Strong','Warm','Same','Yes'],
        ['Sunny','Warm','High','Strong','Warm','Same','Yes'],
        ['Rainy','Cold','High','Strong','Warm','Change','No'],
        ['Sunny','Warm','High','Strong','Cool','Change','Yes']]

target = ['Sky','AirTemp','Humidity','Wind','Water','Forecast','EnjoySport']

df = pd.DataFrame(data,columns=target)

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

def find_s_algorithm(X,y):
    hypothesis = None

    for i,val in enumerate(y):
        if val == "Yes":
            hypothesis = X[i].copy()
            break

    for i,val in enumerate(y):
        if val == "Yes":
            for j in range(len(hypothesis)):
                if hypothesis[j] != X[i][j]:
                    hypothesis[j] = "?"
    return hypothesis


hypothesis = find_s_algorithm(X,y)
print(hypothesis)


        
