#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import sys
from src.cleaning import datacleaning
from src import data_generation
import pandas as pd
import numpy as np
import aif360
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from sklearn.model_selection import train_test_split
from aif360.sklearn.metrics import statistical_parity_difference
from aif360.sklearn.metrics import average_odds_difference
from aif360.sklearn.metrics import equal_opportunity_difference
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


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


# ## Missingness Creation

# We want the attribute with missingness to have around the same proportion of missingness for each type. This is because we don't want the amount of missingness to be a confounding factor in our results.

# In[6]:


mcar = data.copy()
mcar = data_generation.mcar(mcar, 'substantiated')


# In[7]:


mcar['substantiated'].isna().sum() / mcar.shape[0]


# In[8]:


mcar = mcar.dropna(subset = 'substantiated')


# In[9]:


mar = data.copy()
mar = data_generation.mar(mar, 'substantiated', 'complainant_ethnicity', 0.3)


# In[10]:


mar['substantiated'].isna().sum() / mar.shape[0]


# In[11]:


mar = mar.dropna(subset = 'substantiated')


# In[12]:


nmar = data.copy()
nmar = data_generation.nmar(nmar, 'substantiated', 0.3)


# In[13]:


nmar['substantiated'].isna().sum() / nmar.shape[0]


# In[14]:


nmar = nmar.dropna(subset = 'substantiated')


# Now we will "handle" the missingness by dropping missing values.

# ## Applying Fairness Notions
# 

# In[15]:


nypd = data.dropna()


# In[16]:


cat = ["complainant_ethnicity", "complainant_gender", "complainant_age_incident", "allegation", "contact_reason"]


# In[17]:


#model for finding fairness notions when no NaN values are not present in the data
def model(train, test, cats):
    ohe = OneHotEncoder(handle_unknown='ignore')
    
    traincat_df = train[cats]
    # OHE train categorical
    train_ohe = ohe.fit_transform(traincat_df)
    # concat non-cat train features
    train_len = train.shape[0]

    train_num_feats = np.concatenate(
        [np.reshape(train.complainant_age_incident.values, (train_len, 1))
        ], axis = 1
    )
    
    # concatenate train OHE features with non-cat features
    train_feats = pd.DataFrame(np.concatenate([train_ohe.todense(), train_num_feats], axis = 1))
    train_feats['complainant_ethnicity'] = (train['complainant_ethnicity'] == "White").tolist()
    train_feats['complainant_gender'] = (train['complainant_gender'] == "Male").tolist()
    y_train = train.substantiated.values.astype('int')
    
    mod = LogisticRegression(C = 1.0, class_weight='balanced')
    mod.fit(train_feats, y_train)
    
    testcat_df = test[cats]
    # OHE train categorical
    test_ohe = ohe.transform(testcat_df)
    # concat non-cat train features
    test_len = test.shape[0]

    test_num_feats = np.concatenate(
        [np.reshape(test.complainant_age_incident.values, (test_len, 1)),
        ], axis = 1
    )
        
    # concatenate test OHE features with non-cat features
    test_feats = pd.DataFrame(np.concatenate([test_ohe.todense(), test_num_feats], axis = 1))
    test_feats['complainant_ethnicity'] = (test['complainant_ethnicity'] == "White").tolist()
    test_feats['complainant_gender'] = (test['complainant_gender'] == "Male").tolist()
    y_test = test.substantiated.values.astype('int')
    
    pred = mod.predict(test_feats)
    
    parity = statistical_parity_difference(pd.Series(y_test), pd.Series(pred))
    odds = average_odds_difference(pd.Series(y_test), pd.Series(pred))
    opportunity = equal_opportunity_difference(pd.Series(y_test), pd.Series(pred))
    
    
    print("statistical parity: " + str(parity))
    print("Equality of odds: " + str(odds))
    print("Equality of opportunity: " + str(opportunity))
    
    
    return [parity,odds,opportunity]


# In[18]:


#model for finding fairness notions when no NaN values are present in the data
def model_missing(train, test, cats):
    ohe = OneHotEncoder(handle_unknown='ignore')
    
    traincat_df = train[cats]
    # OHE train categorical
    train_ohe = ohe.fit_transform(traincat_df)
    # concat non-cat train features
    train_len = train.shape[0]

    train_num_feats = np.concatenate(
        [np.reshape(train.complainant_age_incident.values, (train_len, 1))
        ], axis = 1
    )
    
    # concatenate train OHE features with non-cat features
    train_feats = pd.DataFrame(np.concatenate([train_ohe.todense(), train_num_feats], axis = 1))
    train_feats['complainant_ethnicity'] = (train['complainant_ethnicity'] == "White").tolist()
    train_feats['complainant_gender'] = (train['complainant_gender'] == "Male").tolist()
    y_train = train.substantiated.values.astype('int')
    
    mod = HistGradientBoostingClassifier()
    mod.fit(train_feats, y_train)
    
    testcat_df = test[cats]
    # OHE train categorical
    test_ohe = ohe.transform(testcat_df)
    # concat non-cat train features
    test_len = test.shape[0]

    test_num_feats = np.concatenate(
        [np.reshape(test.complainant_age_incident.values, (test_len, 1)),
        ], axis = 1
    )
        
    # concatenate test OHE features with non-cat features
    test_feats = pd.DataFrame(np.concatenate([test_ohe.todense(), test_num_feats], axis = 1))
    test_feats['complainant_ethnicity'] = (test['complainant_ethnicity'] == "White").tolist()
    test_feats['complainant_gender'] = (test['complainant_gender'] == "Male").tolist()
    y_test = test.substantiated.values.astype('int')
    
    pred = mod.predict(test_feats)
    
    parity = statistical_parity_difference(pd.Series(y_test), pd.Series(pred))
    odds = average_odds_difference(pd.Series(y_test), pd.Series(pred))
    opportunity = equal_opportunity_difference(pd.Series(y_test), pd.Series(pred))
    
    
    print("statistical parity: " + str(parity))
    print("Equality of odds: " + str(odds))
    print("Equality of opportunity: " + str(opportunity))
    
    
    return [parity,odds,opportunity]


# ### Calculating fairnes notions for No Missingness At All

# In[19]:


train, test = train_test_split(nypd, test_size=0.2)


# In[20]:


#storing fairness notions for no missingness
no_missing_fairness = []
no_missing = model(train, test, cat)
no_missing_fairness.append(no_missing)


# ### Fairness notions for NMAR

# In[21]:


train_nmar, test_nmar = train_test_split(nmar, test_size=0.2)


# In[22]:


nmar_fairness = []
nmar_model = model_missing(train_nmar, test_nmar, cat)
nmar_fairness.append(nmar_model)


# ### Fairness notions for MCAR

# In[23]:


train_mcar, test_mcar = train_test_split(mcar, test_size=0.2)


# In[24]:


mcar_fairness = []
mcar_model = model_missing(train_mcar, test_mcar, cat)
mcar_fairness.append(mcar_model)


# ### Fairness notions for MAR

# In[25]:


train_mar, test_mar = train_test_split(mar, test_size=0.2)


# In[26]:


mar_fairness = []
mar_model = model_missing(train_mar, test_mar, cat)
mar_fairness.append(mar_model)


# In[27]:


#put our fairness statistics into arrays for future usage
par= [no_missing_fairness[0][0],nmar_fairness[0][0],mcar_fairness[0][0],mar_fairness[0][0]]
odds= [no_missing_fairness[0][1],nmar_fairness[0][1],mcar_fairness[0][1],mar_fairness[0][1]]
opp = [no_missing_fairness[0][2],nmar_fairness[0][2],mcar_fairness[0][2],mar_fairness[0][2]]


# ## Visualizing Our Results

# In[28]:


import matplotlib.pyplot as plt


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


# In[ ]:




