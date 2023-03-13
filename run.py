#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
from src.cleaning import datacleaning
from src import data_generation
from src.preparation import model_perform
import pandas as pd
import numpy as np
import aif360
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from aif360.sklearn.metrics import statistical_parity_difference
from aif360.sklearn.metrics import average_odds_difference
from aif360.sklearn.metrics import equal_opportunity_difference


# In[2]:


data = datacleaning.cleaning(os.path.join(os.path.dirname(
    os.path.realpath('run.py')) + '/data/allegations_raw.csv'))


# In[3]:


target = sys.argv[1]

if target == "test":
    data = datacleaning.cleaning(os.path.join(os.path.dirname(
        os.path.realpath('run.py')) + '/data/test.csv'))

if target == "all":
    data = datacleaning.cleaning(os.path.join(os.path.dirname(
        os.path.realpath('run.py')) + '/data/allegations_raw.csv'))


# In[4]:


data.isna().sum()


# In[5]:


data.head()


# In[6]:


train, test = train_test_split(data, train_size=0.8)


# ## Missingness Creation

# We want the attribute with missingness to have around the same proportion of missingness for each type. This is because we don't want the amount of missingness to be a confounding factor in our results.

# In[7]:


t = train.copy()
mcar = data_generation.mcar(t, 'complainant_gender')


# In[8]:


mcar['complainant_gender'].isna().sum() / mcar.shape[0]


# In[10]:


mcar = mcar.dropna(subset = 'complainant_gender')


# For MAR, we want to try to find attributes that are, ideally, highly correlated with both the sensitive attribute and the label outcome attribute. Let's first look into the complainant age attribute.

# In[11]:


for gender in ['Male', 'Female']:
    for sub in [True, False]:
        data[(data['complainant_gender'] == gender) & (
            data['substantiated'] == sub)]['complainant_age_incident'].plot(kind='density', legend=True)
    plt.legend(['MaleSub', 'MaleNot', 'FemaleSub', 'FemaleNot'])


# It looks like complainant age is only highly correlated with the sensitive attribute, and not the outcome label. This will still work with the goal we have in mind, though. 

# In[12]:


for gender in ['Male', 'Female']:
    data[(data['complainant_gender'] == gender)]['complainant_age_incident'].plot(kind='density', legend=True)
plt.legend(['MaleSub', 'MaleNot', 'FemaleSub', 'FemaleNot'])


# In order to add to the noticeability of MAR (where each category of our sensitive attribute has a different amount of missinness) we will only apply MAR missingness to the underprivileged group (females). Non-MAR missingness can be added to the privileged group to see how different types of missingness produce differences among the sensitive attribute.

# In[13]:


t = train.copy()
male = data_generation.mcar(t[t['complainant_gender'] == 'Male'], 'complainant_gender')
female = data_generation.mar(t[t['complainant_gender'] == 'Female'], 'complainant_gender', 'substantiated', 0.2)
for idx, i in enumerate(female.iterrows()):
    # if S = 0 & Y = 1, add additional probability of missingness
    if (i[1]['substantiated'] == True) and (i[1]['complainant_gender'] == 'Female'):
        if np.random.choice([1, 0], p = [0.3, 0.7]) == 1:
            female['complainant_gender'].iloc[idx] = np.nan
mar = pd.concat([male, female])
mar


# In[14]:


#mar = data_generation.mar(t, 'substantiated', 'complainant_gender', 0.3)


# In[15]:


mar['complainant_gender'].isna().sum() / mar.shape[0]


# In[17]:


mar = mar.dropna(subset = 'complainant_gender')


# In[18]:


t = train.copy()
nmar = data_generation.nmar(t, 'complainant_gender', 0.3)


# In[19]:


nmar['complainant_gender'].isna().sum() / nmar.shape[0]


# In[21]:


nmar = nmar.dropna(subset = 'substantiated')


# Now we will "handle" the missingness by dropping missing values.

# ## Applying Fairness Notions
# 

# In[22]:


cats = ["allegation", "contact_reason"]


# ### Calculating fairnes notions for No Missingness At All

# In[23]:


#storing fairness notions for no missingness
no_missing_results = model_perform.model(train, test, cats)


# ### Fairness notions for NMAR

# In[24]:


nmar_results = model_perform.model(nmar, test, cats)


# ### Fairness notions for MCAR

# In[25]:


mcar_results = model_perform.model(mcar, test, cats)


# ### Fairness notions for MAR

# In[26]:


mar_results = model_perform.model(mar, test, cats)


# In[27]:


nmar.head()


# ## Visualizing Our Results

# In[28]:


#put our fairness statistics into arrays for future usage
acc = [no_missing_results[0],nmar_results[0],mcar_results[0],mar_results[0]]
par= [no_missing_results[1],nmar_results[1],mcar_results[1],mar_results[1]]
odds= [no_missing_results[2],nmar_results[2],mcar_results[2],mar_results[2]]
opp = [no_missing_results[3],nmar_results[3],mcar_results[3],mar_results[3]]


# In[29]:


labels = ['No Missingess,', 'NMAR', 'MCAR', 'MAR']


# In[30]:


plt.figure(figsize = (20, 10))
plt.title('Statistical Parities')
plt.xlabel('Missingness Type')
plt.ylabel('Statistical Parotities')
plt.ylim(min(par) - 0.1, max(par) + 0.1)
plt.plot(labels, par, marker='.', markersize = 20)


# In[31]:


plt.figure(figsize = (20, 10))
plt.title('Equality of Odds')
plt.xlabel('Missingness Type')
plt.ylabel('Equality of Odds')
plt.ylim(min(odds)-0.1,max(odds) + 0.1)
plt.plot(labels, odds, marker='.', markersize = 20)


# In[32]:


plt.figure(figsize = (20, 10))
plt.title('Equality of Opportunity')
plt.xlabel('Missingness Type')
plt.ylabel('Equality of Opportunity')
plt.ylim(min(opp)-0.1,max(opp) + 0.1)
plt.plot(labels, opp, marker='.', markersize = 20)


# In[33]:


plt.figure(figsize = (20, 10))
plt.title('Accuracies')
plt.xlabel('Missingness Type')
plt.ylabel('Accuracies')
plt.ylim(min(acc)-0.1,max(acc) + 0.1)
plt.plot(labels, acc, marker='.', markersize = 20)


# In[ ]:




