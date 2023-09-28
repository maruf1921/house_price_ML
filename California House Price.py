#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install pandas numpy matplotlib seaborn


# # Loading data set

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv("housing.csv")



# # Data exploration

# In[4]:


data.info()


# In[5]:


data.dropna(inplace=True)


# In[6]:


data.info()


# In[7]:


data.hist(figsize = (15,8))


# In[8]:


plt.figure(figsize=(15,8))
sns.heatmap(data.corr(), annot = True, cmap = 'YlGnBu' )


# # Data preprocessing

# In[9]:


data['total_rooms'] = np.log(data['total_rooms']+1)
data['total_bedrooms'] = np.log(data['total_bedrooms']+1)
data['households'] = np.log(data['households']+1)
data['population'] = np.log(data['population']+1)


# In[10]:


data.hist(figsize = (15,8))


# In[11]:


data.ocean_proximity.value_counts()


# In[12]:


pd.get_dummies(data.ocean_proximity)


# In[13]:


data = data.join(pd.get_dummies(data.ocean_proximity)).drop(['ocean_proximity'], axis = 1)


# In[14]:





# In[15]:


plt.figure(figsize=(15,8))
sns.heatmap(data.corr(), annot = True, cmap = 'YlGnBu' )


# In[16]:


plt.figure(figsize=(15,8))
sns.scatterplot(x = "latitude", y = "longitude", data= data, hue = "median_house_value", palette="coolwarm")


# # Feature Engineering

# In[17]:


data['bedroom_ratio'] = data['total_bedrooms']/data['total_rooms']
data['households_rooms'] = data['total_rooms']/data['households']


# In[18]:


plt.figure(figsize=(15,8))
sns.heatmap(data.corr(), annot = True, cmap = 'YlGnBu' )


# # Data splitting

# In[19]:


from sklearn.model_selection import train_test_split

X = data.drop(['median_house_value'], axis = 1)
y = data['median_house_value']


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[21]:


train_data = X_train.join(y_train)
test_data  = X_test.join(y_test)


# # Linear Regression Model

# In[22]:


from sklearn.linear_model import LinearRegression



reg = LinearRegression()

reg.fit(X_train, y_train)


# In[23]:


reg.score(X_test, y_test)


# # Prediction

# In[24]:


# Create a dictionary with input data in the same format as your training data
# These values represent features of the neighborhood where you want to predict the median house value.

input_data = {
    'longitude': [-122.23],  # Replace with the longitude of your input
    'latitude': [37.88],     # Replace with the latitude of your input
    'housing_median_age': [41],  # Replace with the housing median age of your input
    'total_rooms': [880],     # Replace with the total rooms of your input
    'total_bedrooms': [129],  # Replace with the total bedrooms of your input
    'population': [322],      # Replace with the population of your input
    'households': [126],      # Replace with the households of your input
    'median_income': [8.3252],  # Replace with the median income of your input
    'median_income': [12.3886],     # Replace with the actual median income of the people in the neighborhood
    '<1H OCEAN': [1],               # This indicates whether the neighborhood is close to the ocean (0 for No, 1 for Yes)
    'INLAND': [0],                  # This indicates whether the neighborhood is inland (0 for No, 1 for Yes)
    'ISLAND': [0],                  # This indicates whether the neighborhood is on an island (0 for No, 1 for Yes)
    'NEAR BAY': [0],                # This indicates whether the neighborhood is near a bay (0 for No, 1 for Yes)
    'NEAR OCEAN': [0]               # This indicates whether the neighborhood is near the ocean (0 for No, 1 for Yes)
}


# Create a DataFrame from the input data
input_df = pd.DataFrame(input_data)

# Preprocess the input data in the same way as your training data
input_df['total_rooms'] = np.log(input_df['total_rooms'] + 1)
input_df['total_bedrooms'] = np.log(input_df['total_bedrooms'] + 1)
input_df['households'] = np.log(input_df['households'] + 1)
input_df['population'] = np.log(input_df['population'] + 1)
#input_df = input_df.join(pd.get_dummies(input_df['ocean_proximity'])).drop(['ocean_proximity'], axis=1)
input_df['bedroom_ratio'] = input_df['total_bedrooms'] / input_df['total_rooms']
input_df['households_rooms'] = input_df['total_rooms'] / input_df['households']

# Make the prediction using the trained model
predicted_value = reg.predict(input_df)

# Print the predicted median house value
print("Predicted Median House Value:", predicted_value[0])

