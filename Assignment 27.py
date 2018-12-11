
# coding: utf-8

# Problem Statement
# 
# In this assignment, students will be using the K-nearest neighbors algorithm to predict how many points NBA players scored in the 2013-2014 season.
# 
# A look at the data
# 
# Before we dive into the algorithm, let’s take a look at our data. Each row in the data contains information on how a player performed in the 2013-2014 NBA season.
# 
# Download 'nba_2013.csv' file from this link:
# https://www.dropbox.com/s/b3nv38jjo5dxcl6/nba_2013.csv?dl=0
# 
# Here are some selected columns from the data:
# player - name of the player
# pos - the position of the player
# g - number of games the player was in
# gs - number of games the player started
# pts - total points the player scored
# 
# There are many more columns in the data, mostly containing information about average player game performance over the course of the season. See this site for an explanation of the rest of them.
# 
# We can read our dataset in and figure out which columns are present:

# In[1]:


# importing important library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
with open("nba_2013.csv", 'r') as csvfile:
    nba = pd.read_csv(csvfile)


# In[2]:


nba.head()


# In[3]:


# data info
nba.info()


# In[4]:


# columns name
nba.columns


# In[5]:


# checking for null value
nba.isnull().sum()


# In[6]:


# droping columns having null values
nba.dropna(inplace=True)


# In[7]:


# checking for null values
nba.isnull().sum()


# In[8]:


# checking df shape
nba.shape


# In[9]:


# making a new data frame x

x=nba[['g', 'gs', 'mp', 'fg', 'fga',
       'fg.', 'x3p', 'x3pa', 'x3p.', 'x2p', 'x2pa', 'x2p.', 'efg.', 'ft',
       'fta', 'ft.', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf',
      ]]
x.head()


# In[10]:


x.info()


# In[11]:


# making new data frame y
y=nba[['pts']]
y.head()


# Step 1:- We will train the model without doing any data scaling

# In[12]:


# importing the libraries
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# dividing into train-test set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=42)

# printing the shape of the data frame
print("x_train shape:",x_train.shape)
print("x_test shape:",x_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:",y_test.shape)


# In[13]:


# ignoring the warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score

#finding the best k value using cross validation
error=[]
for i in range(1,30,1):
    knn=KNeighborsRegressor(n_neighbors=i);
    scores=cross_val_score(knn,x_train,y_train,cv=10,scoring='mean_squared_error')
    error.append(scores.mean())
error[:5]


# In[14]:


x_error=[i for i in range(1,30,1)]


# In[15]:


# plotting error vs k value
plt.plot(x_error,error)


# In[16]:


# model training and prediction on optimal k value
regressor=KNeighborsRegressor(n_neighbors=6)
regressor.fit(x_train,y_train)
y_predict=regressor.predict(x_test)
error=mean_squared_error(y_test,y_predict)
error


# Step 2:- We will train the model after doing the normalization

# In[17]:


x.head()


# In[18]:


# normalizing the data of the data frame
x_norm=(x-x.min())/(x.max()-x.min())
x_norm.head()


# In[19]:


# training test split
x_train,x_test,y_train,y_test=train_test_split(x_norm,y,test_size=0.25, random_state=42)

# printing the shape 
print("x_train shape:",x_train.shape)
print("x_test shape:",x_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:",y_test.shape)


# In[20]:


# filtering the warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score

# finding optimal k value using cross validation
error=[]
for i in range(1,30,1):
    knn=KNeighborsRegressor(n_neighbors=i);
    scores=cross_val_score(knn,x_train,y_train,cv=10,scoring='mean_squared_error')
    error.append(scores.mean())
error[:5]


# In[21]:


x_error=[i for i in range(1,30,1)]


# In[22]:


# plotting error vs k value
plt.plot(x_error,error)


# In[23]:


# training and predicting model on the optimal k value
regressor=KNeighborsRegressor(n_neighbors=4)
regressor.fit(x_train,y_train)
y_predict=regressor.predict(x_test)
error=mean_squared_error(y_test,y_predict)
error


# Step 3:- We will train the model after doing the standardization

# In[24]:


x.head()


# In[25]:


# standardizing value by using below formula a type of scaling
x_stand=(x-x.mean())/x.std()
x_stand.head()


# In[26]:


# dividing the model into train and test
x_train,x_test,y_train,y_test=train_test_split(x_stand,y,test_size=0.25, random_state=42)

#printing the shape
print("x_train shape:",x_train.shape)
print("x_test shape:",x_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:",y_test.shape)


# In[27]:


# ignoring the warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score

# finding optimal k value by using cross validation
error=[]
for i in range(1,30,1):
    knn=KNeighborsRegressor(n_neighbors=i);
    scores=cross_val_score(knn,x_train,y_train,cv=10,scoring='mean_squared_error')
    error.append(scores.mean())
error[:5]


# In[28]:


x_error=[i for i in range(1,30,1)]


# In[29]:


# plotting graph between error vs k value
plt.plot(x_error,error)


# In[30]:


# training model and predicting value on the optimal k value
regressor=KNeighborsRegressor(n_neighbors=5)
regressor.fit(x_train,y_train)
y_predict=regressor.predict(x_test)
error=mean_squared_error(y_test,y_predict)
error


# Conclusion
# 
# For data without any scaling Mean squared error =4938.488723872388 , at k=6
# 
# For nornalized data Mean squared error =6814.95853960396 , at k=4
# 
# For standardized data Mean squared error = 9396.441188118812 , at k=5
