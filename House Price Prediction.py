#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt


# In[2]:


df = pd.read_csv('Real estate.csv')


# In[3]:


# Display summary statistics
print(df.describe())


# In[4]:


# Histograms for each numerical feature
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()


# In[5]:


# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# In[6]:


X = df.drop(columns=['Y house price of unit area'])
Y = df['Y house price of unit area']


# In[7]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[8]:


model = LinearRegression()
model.fit(X_train, Y_train)


# In[9]:


Y_pred = model.predict(X_test)


# In[10]:


mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = sqrt(mse)


# In[11]:


print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)


# In[ ]:




