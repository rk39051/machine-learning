'''
Created on 11-Jun-2020

@author: Rupesh
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#pd.set_option('max_column',None)
#pd.set_option('max_row',None)


#loading the data and viewing the shape
data1=pd.read_csv('C:\\Users\\Rupesh\\Downloads\\train.csv')
data2=pd.read_csv('C:\\Users\\Rupesh\\Downloads\\test.csv')
y=data1["SalePrice"] #load the dependent variable
data1=data1.drop(columns=["SalePrice"])
dataset=pd.concat([data1,data2])
print(dataset.head(2))

print(dataset.shape)

#remove some unwanted columns from dataset
unwanted_columns=[]
for i in dataset.columns:
    if (dataset[i].isnull().sum()>=0.999*len(dataset) or i=="Id"):
        unwanted_columns.append(i)


print("printing unwanted columns\n ",unwanted_columns)   

dataset=dataset.drop(columns=unwanted_columns)

#extracting categorical, numerical and temporal variabe(date time year)
categorial_features=[]
numerical_features=[]
temporal_features=[]


for i in dataset.columns:
    if "Year" in i or "Yr" in i:
        temporal_features.append(i)
        
for i in dataset.columns:
    if dataset[i].dtype!='O':
        numerical_features.append(i)
    if dataset[i].dtype =='O' and i not in (temporal_features):
        categorial_features.append(i)


print("printing temporal features \n",dataset[temporal_features].head(2))


print("printing categorical features \n",dataset[categorial_features].head(5))   


#dealing with categorical missing value
def categorical_missed_value(dataset,feature):
    data=dataset.copy()
    data[feature]=data[feature].fillna("missing")
    return data
        
dataset=categorical_missed_value(dataset,categorial_features)  


#dealing with numerical missing value
   
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
dataset[numerical_features] = imputer.fit_transform(dataset[numerical_features])



#encoding categorical variable
dataset=pd.get_dummies(drop_first=True,data=dataset,columns=categorial_features)

print(dataset.head(5))       

#dividing the labeled data and unlabeled data
train_data,test_data=dataset.iloc[:1460,:],dataset.iloc[1460:,:]

#split the data labeled data into train and test
X=train_data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)



# Fit Random Forest Regression to the dataset
print("model is being developed \n")
from sklearn.ensemble import RandomForestRegressor 
rfr = RandomForestRegressor(n_estimators =400)#n_estimator is no of decision tree
rfr.fit(X_train, y_train)


#test your model on X_test (splitted dataset)
y_pred=rfr.predict(X_test)

print("printing actual SalePrice value\n",y_test)
print("printing pridicted SalePrice value \n",y_pred)


#predict the result of test_data for submission
result=rfr.predict(test_data)
print("printing SalePrice pridiction for unlabeled dataset \n",result)

for i in data2.columns:
    if i !="Id":
        data2=data2.drop(columns=[i])

result=pd.DataFrame(result) 
result=result.rename(columns={0:"SalePrice"}) 

final=pd.concat([data2,result],axis=1)
print("printing 10 final results\n",final.head(10))

      
