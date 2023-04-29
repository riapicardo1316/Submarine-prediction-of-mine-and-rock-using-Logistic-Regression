#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[ ]:


# loading the dataset
from pandas import read_csv
sonar_data=pd.read_csv('Copy of sonar data.csv',header=None)


# In[ ]:


sonar_data.head()


# In[ ]:


# number of rows and columns
sonar_data.shape


# In[ ]:


sonar_data.describe()


# In[ ]:


# counting the number of examples of mines and rocks
sonar_data[60].value_counts()


# In[ ]:


# grouping the mine and rock examples by its mean
sonar_data.groupby(60).mean()


# In[ ]:


# separating the data and labels
X=sonar_data.drop(columns=60,axis=1)
Y=sonar_data[60]


# In[ ]:


print(X)
print(Y)


# In[ ]:


#splitting training and testing data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)


# In[ ]:


print(X.shape,X_train.shape,X_test.shape)
print(Y.shape,Y_train.shape,Y_test.shape)


# In[ ]:


print(X_train)
print(Y_train)


# In[ ]:


#model training
model=LogisticRegression()


# In[ ]:


#training the logistic model
model.fit(X_train,Y_train)


# In[ ]:


#model evaluation
#accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[ ]:


print('Accuracy of training data :',training_data_accuracy)


# In[ ]:


#accuracy on testing data
X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[ ]:


print('Accuracy of testing data :',testing_data_accuracy)


# In[ ]:


#making predictive model
input_data=(0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032)
          
input_data_numpy=np.asarray(input_data)
#since we will be testing with only one instane we have to reshape the data
input_data_reshaped=input_data_numpy.reshape(1,-1)

prediction=model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]=='R'):
    print('The object is ROCK')
else:
    print('The object is MINE')


# In[ ]:




