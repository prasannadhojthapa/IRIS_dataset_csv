#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

#Install Kaggle
get_ipython().system('pip install kaggle')


# **Importing the CSV data:**

# In[86]:


#test1 = pd.read_csv('C:\\Users\\prasa\\Downloads\\Kaggle Files\\Hotel_Reservations_Dataset\\Hotel Reservations.csv',sep ='\t')
#Note: The code below has been working because of the UTF-8. It has rectified the encoding.

#this loads the data in df
df  = pd.read_csv('C:\\Users\\prasa\\Downloads\\Kaggle Files\\Kaggle CSV Files\\IRIS.csv',encoding='UTF-8')
print(df.iloc[:,:])
print()
print("data shape: ", df.shape)
print("data dimension: ", df.ndim)


# In[32]:


df.head()


df.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length','Petal_Width','Species']

X = df.drop("Sepal_Length",axis=1)   #Feature Matrix
Y = df["Sepal_Length"]          #Target Variable


# In[35]:


#f.head()
#This is for the features description:
#rint("Keys of df dataset: \n {}".format(df.keys()))

#rint("Target names: {}".format(df['target_names']))


# In[108]:


#features = df.columns


features = df.iloc[:,0:4]
features_list= list(features)

print(features_list)
print("Shape of the data: {}".format(features.shape))


# In[267]:


#Trial 1

import numpy as np
import csv
from sklearn.utils import Bunch
from sklearn import preprocessing

def load_my_dataset():
    with open('C:\\Users\\prasa\\Downloads\\Kaggle Files\\Kaggle CSV Files\\IRIS.csv') as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        print(type(temp))
        n_samples = 150 #number of data rows, don't count header
        n_features = 4 #number of columns for features, don't count target column
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'] #adjust accordingly
        target_names = ['species'] #adjust accordingly
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)
        
        print(type(data_file))
        for i, sample in enumerate(data_file):
            data[i] = np.asarray(sample[:-1], dtype=np.float64)
            #print(target_names)    
            #Because the target is in strings we need to convert it into integers: Using factorize.
            #target_names = pd.factorize(target_names)[0]
            #target[i] = np.asarray(sample[-1], dtype=np.int)
        #print(data)
    
    return Bunch(data=data, target=target, feature_names = feature_names, target_names = target_names)
#print(target_names)
data = load_my_dataset()


# In[214]:


#Trial 2

# Python program to illustrate
# creating a data frame using CSV files
 
# import pandas module
import pandas as pd
# import csv module
import csv
 
with open("C:\\Users\\prasa\\Downloads\\Kaggle Files\\Kaggle CSV Files\\IRIS.csv") as csv_file:
    # read the csv file
    csv_reader = csv.reader(csv_file)
 
    # now we can use this csv files into the pandas
    df = pd.DataFrame([csv_reader], index = None)
 


# In[338]:


#Trial 3

import pandas as pd
# The file has no headers naming the columns, so we pass header=None
# and provide the column names explicitly in "names"
df = pd.read_csv(
    "C:\\Users\\prasa\\Downloads\\Kaggle Files\\Kaggle CSV Files\\IRIS.csv", header=None, index_col=False,
    names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
# IPython.display allows nice output formatting within the Jupyter notebook
display(df.head())


# In[339]:


# Drop first row using drop()
df.drop(index=df.index[0], axis=0, inplace=True)


# In[340]:


print(df.species.value_counts())

print(df.species)

#assigning numbers for the species values:
assign_species = {'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}
lst = [assign_species[k] for k in df.species]
df = df.values.tolist()


# In[349]:


#To merge the two matrix

import sympy as sp

a = sp.Matrix(df).col_insert(-1, sp.Matrix(lst))
a_ = a.tolist()


# In[350]:


#To pop the str species column

for row in a_:
    del row[-1]  # 0 for column 1, 1 for column 2, etc.

print(a_)


# In[362]:


print(type(a_))
print(type(a_))
an_iterator = iter(a_)
print(an_iterator)


# In[376]:


print(next(an_iterator))
print(next(an_iterator))


# In[383]:


def load_my_dataset():
    #changing the type dataframe to type list for 'df'
    temp = next(an_iterator)
    n_samples = 150 #number of data rows, don't count header
    n_features = 4 #number of columns for features, don't count target column
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'] #adjust accordingly
    target_names = ['species'] #adjust accordingly
    data = np.empty((n_samples, n_features))
    target = np.empty((n_samples,), dtype=np.int64)
    
    for i, sample in enumerate(a_):
        data[i] = np.asarray(sample[:-1], dtype=np.float64)
        #print(target_names)    
        #Because the target is in strings we need to convert it into integers: Using factorize.
        #target_names = pd.factorize(target_names)[0]
        target[i] = np.asarray(sample[-1], dtype=np.int64)
        #print(data)
    
    return Bunch(data=data, target=target, feature_names = feature_names, target_names = target_names)
data = load_my_dataset()


# In[392]:


print(data.target)


# In[393]:


#converting bunch object to panda dataframe:
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = pd.Series(data.target)
#print(np.random.shuffle(data))


# In[406]:


#print(df)
print(data.target)
print(df.shape)
print(df.ndim)


# **Train Test Split**

# In[404]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)


# In[407]:


print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

print()
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))


# #### Plotting
# 

# In[413]:


#Installing mglearn

get_ipython().system('pip install mglearn')


# In[415]:


from pandas.plotting import scatter_matrix

import mglearn

# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=data.feature_names)
# create a scatter matrix from the dataframe, color by y_train
grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)


# #### KNeighborsClassifier
# 

# In[417]:


from sklearn.neighbors import KNeighborsClassifier
#Instantiate
knn = KNeighborsClassifier(n_neighbors=1)

#To build the model on the training set, call `fit` method of the knn object
knn.fit(X_train, y_train)


# Scikit-learn always expects two-dimensional arrays for data.

# In[420]:


y_pred = knn.predict(X_test)
print("Test set predictions: \n{}".format(y_pred))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


# In[ ]:




