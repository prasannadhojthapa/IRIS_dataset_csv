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


# In[215]:


#Trial 1

import numpy as np
import csv
from sklearn.utils import Bunch
from sklearn import preprocessing

def load_my_dataset():
    with open('C:\\Users\\prasa\\Downloads\\Kaggle Files\\Kaggle CSV Files\\IRIS.csv') as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        print(temp)
        n_samples = 150 #number of data rows, don't count header
        n_features = 4 #number of columns for features, don't count target column
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'] #adjust accordingly
        target_names = ['species'] #adjust accordingly
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)
                
        #print(data)
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
 


# In[225]:


#Trial 3

import pandas as pd
# The file has no headers naming the columns, so we pass header=None
# and provide the column names explicitly in "names"
data = pd.read_csv(
    "C:\\Users\\prasa\\Downloads\\Kaggle Files\\Kaggle CSV Files\\IRIS.csv", header=None, index_col=False,
    names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
# IPython.display allows nice output formatting within the Jupyter notebook
display(data.head())


# In[234]:


print(data.species[1:].value_counts())
print(data[1:])


# In[ ]:




