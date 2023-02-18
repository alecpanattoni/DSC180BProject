#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import sys
from src.cleaning import datacleaning
from src import data_generation
from src.preparation import model_perform
import pandas as pd
import numpy as np
import aif360
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
mcar = data_generation.mcar(t, 'substantiated')


# In[8]:


mcar['substantiated'].isna().sum() / mcar.shape[0]


# In[9]:


mcar = mcar.dropna(subset = 'substantiated')


# In[10]:


t = train.copy()
mar = data_generation.mar(t, 'substantiated', 'complainant_ethnicity', 0.3)


# In[11]:


mar['substantiated'].isna().sum() / mar.shape[0]


# In[12]:


mar = mar.dropna(subset = 'substantiated')


# In[13]:


t = train.copy()
nmar = data_generation.nmar(t, 'substantiated', 0.3)


# In[14]:


nmar['substantiated'].isna().sum() / nmar.shape[0]


# In[15]:


nmar = nmar.dropna(subset = 'substantiated')


# Now we will "handle" the missingness by dropping missing values.

# ## Applying Fairness Notions
# 

# In[16]:


cat = ["complainant_ethnicity", "complainant_age_incident", "allegation", "contact_reason"]


<<<<<<< HEAD
# In[17]:


acc = []


# ### Calculating fairnes notions for No Missingness At All

# In[18]:
=======
# ### Calculating fairnes notions for No Missingness At All

# In[17]:
>>>>>>> ea230a069629d60e2d13335a75246535f4f5af9d


#storing fairness notions for no missingness
no_missing_fairness = []
no_missing = model_perform.model(train, test, cat)
no_missing_fairness.append(no_missing)


<<<<<<< HEAD
# In[19]:


no_missing


# ### Fairness notions for NMAR

# In[20]:
=======
# ### Fairness notions for NMAR

# In[18]:
>>>>>>> ea230a069629d60e2d13335a75246535f4f5af9d


train_nmar, test_nmar = train_test_split(nmar, test_size=0.2)


<<<<<<< HEAD
# In[21]:
=======
# In[19]:
>>>>>>> ea230a069629d60e2d13335a75246535f4f5af9d


nmar_fairness = []
nmar_model = model_perform.model_missing(train_nmar, test_nmar, cat)
nmar_fairness.append(nmar_model)


# ### Fairness notions for MCAR

<<<<<<< HEAD
# In[22]:
=======
# In[20]:
>>>>>>> ea230a069629d60e2d13335a75246535f4f5af9d


train_mcar, test_mcar = train_test_split(mcar, test_size=0.2)


<<<<<<< HEAD
# In[23]:
=======
# In[21]:
>>>>>>> ea230a069629d60e2d13335a75246535f4f5af9d


mcar_fairness = []
mcar_model = model_perform.model_missing(train_mcar, test_mcar, cat)
mcar_fairness.append(mcar_model)


# ### Fairness notions for MAR

<<<<<<< HEAD
# In[24]:
=======
# In[22]:
>>>>>>> ea230a069629d60e2d13335a75246535f4f5af9d


train_mar, test_mar = train_test_split(mar, test_size=0.2)


<<<<<<< HEAD
# In[25]:
=======
# In[23]:
>>>>>>> ea230a069629d60e2d13335a75246535f4f5af9d


mar_fairness = []
mar_model = model_perform.model_missing(train_mar, test_mar, cat)
mar_fairness.append(mar_model)


<<<<<<< HEAD
# In[26]:


#put our fairness statistics into arrays for future usage
acc = [no_missing_fairness[0][0],nmar_fairness[0][0],mcar_fairness[0][0],mar_fairness[0][0]]
par= [no_missing_fairness[0][1],nmar_fairness[0][1],mcar_fairness[0][1],mar_fairness[0][1]]
odds= [no_missing_fairness[0][2],nmar_fairness[0][2],mcar_fairness[0][2],mar_fairness[0][2]]
opp = [no_missing_fairness[0][3],nmar_fairness[0][3],mcar_fairness[0][3],mar_fairness[0][3]]
=======
# In[24]:


#put our fairness statistics into arrays for future usage
par= [no_missing_fairness[0][0],nmar_fairness[0][0],mcar_fairness[0][0],mar_fairness[0][0]]
odds= [no_missing_fairness[0][1],nmar_fairness[0][1],mcar_fairness[0][1],mar_fairness[0][1]]
opp = [no_missing_fairness[0][2],nmar_fairness[0][2],mcar_fairness[0][2],mar_fairness[0][2]]
>>>>>>> ea230a069629d60e2d13335a75246535f4f5af9d


# ## Visualizing Our Results

<<<<<<< HEAD
# In[27]:
=======
# In[25]:
>>>>>>> ea230a069629d60e2d13335a75246535f4f5af9d


import matplotlib.pyplot as plt


<<<<<<< HEAD
# In[28]:
=======
# In[26]:
>>>>>>> ea230a069629d60e2d13335a75246535f4f5af9d


labels = ['No Missingess,', 'NMAR', 'MCAR', 'MAR']


<<<<<<< HEAD
# In[29]:
=======
# In[27]:
>>>>>>> ea230a069629d60e2d13335a75246535f4f5af9d


plt.figure(figsize = (20, 10))
plt.title('Statistical Parities')
plt.xlabel('Missingness Type')
plt.ylabel('Statistical Parotities')
plt.ylim(min(par) - 0.1, max(par) + 0.1)
plt.plot(labels, par, marker='.', markersize = 20)


<<<<<<< HEAD
# In[30]:
=======
# In[28]:
>>>>>>> ea230a069629d60e2d13335a75246535f4f5af9d


plt.figure(figsize = (20, 10))
plt.title('Equality of Odds')
plt.xlabel('Missingness Type')
plt.ylabel('Equality of Odds')
plt.ylim(min(odds)-0.1,max(odds) + 0.1)
plt.plot(labels, odds, marker='.', markersize = 20)


<<<<<<< HEAD
# In[31]:
=======
# In[29]:
>>>>>>> ea230a069629d60e2d13335a75246535f4f5af9d


plt.figure(figsize = (20, 10))
plt.title('Equality of Opportunity')
plt.xlabel('Missingness Type')
plt.ylabel('Equality of Opportunity')
plt.ylim(min(opp)-0.1,max(opp) + 0.1)
plt.plot(labels, opp, marker='.', markersize = 20)


<<<<<<< HEAD
# In[33]:


plt.figure(figsize = (20, 10))
plt.title('Accuracies')
plt.xlabel('Missingness Type')
plt.ylabel('Accuracies')
plt.ylim(min(acc)-0.1,max(acc) + 0.1)
plt.plot(labels, acc, marker='.', markersize = 20)


=======
>>>>>>> ea230a069629d60e2d13335a75246535f4f5af9d
# In[ ]:




