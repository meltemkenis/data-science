#!/usr/bin/env python
# coding: utf-8

# In[40]:


import warnings
warnings.filterwarnings('ignore')


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


file_name='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)


# In[8]:


df.head()


# In[9]:


df.dtypes


# In[10]:


df.describe()


# In[11]:


df.drop(["id", "Unnamed: 0"], axis=1, inplace=True)


# In[12]:


df.describe()


# In[13]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# In[14]:


mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)


# In[15]:


mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)


# In[16]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# In[17]:


floors_counts = df["floors"].value_counts()
floors_counts.to_frame()


# In[18]:


sns.boxplot(x="waterfront", y="price",data=df);


# In[19]:


sns.regplot(x='sqft_above', y="price", data=df);
plt.ylim(0,);


# In[20]:


df.corr()['price'].sort_values()


# In[21]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[22]:


X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm
lm.fit(X,Y)
lm.score(X, Y)


# In[23]:


X = df[['sqft_living']]
Y = df['price']
lm = LinearRegression()
lm
lm.fit(X,Y)
lm.score(X, Y)


# In[29]:


features =df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]   


# In[30]:


lm = LinearRegression()
lm
lm.fit(features,Y)
lm.score(features, Y)


# In[31]:


Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


# In[32]:


pipe=Pipeline(Input)
pipe


# In[41]:


pipe.fit(features,Y)


# In[42]:


pipe.score(features,Y)


# In[44]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# In[49]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features ]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# In[46]:


from sklearn.linear_model import Ridge


# In[55]:


RidgeModel = Ridge(alpha = 0.1)
RidgeModel.fit(x_train,y_train)
Yhat = RidgeModel.predict(x_test)


# In[59]:


RidgeModel.score(x_test, y_test)


# In[68]:


from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree = 2)

x_train_polly = pr.fit_transform(x_train)
x_test_polly =  pr.fit_transform(x_test)


# In[69]:


RidgeModel2 = Ridge(alpha = 0.1)
RidgeModel2.fit(x_train_polly,y_train)
Yhat = RidgeModel.predict(x_test_polly)


# In[70]:


RidgeModel2.score(x_test_polly, y_test)


# In[ ]:




