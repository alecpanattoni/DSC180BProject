#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import sys
from src import data_generation
from src.cleaning import datacleaning

import aif360
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from sklearn.model_selection import train_test_split
from aif360.sklearn.metrics import statistical_parity_difference
from aif360.sklearn.metrics import average_odds_difference
from aif360.sklearn.metrics import equal_opportunity_difference
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.sklearn.metrics import disparate_impact_ratio, average_odds_error, generalized_fpr
from aif360.sklearn.metrics import generalized_fnr, difference

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from sklearn.preprocessing import OneHotEncoder, StandardScaler

# nypd = pd.read_csv("https://raw.githubusercontent.com/IBM/AIF360/master/examples/data/compas/compas-scores-two-years.csv")

data = datacleaning.cleaning(os.path.join(os.path.dirname(
    os.path.realpath('run.py')) + '/data/allegations_raw.csv'))

def model(train, test, cats):
    
    # Encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore')
    train_enc = pd.DataFrame(encoder.fit_transform(train[cats]).toarray())
    test_enc = pd.DataFrame(encoder.transform(test[cats]).toarray())
    # still need to account for substantiated, age, and gender
    other_train = train[['complainant_age_incident', 'complainant_gender', 'substantiated']]
    other_test = test[['complainant_age_incident', 'complainant_gender', 'substantiated']]
    other_train.index = range(len(train_enc))
    other_test.index = range(len(test_enc))
    # concatenate OHE features with other features
    train = pd.concat([train_enc, other_train], axis = 1)
    test = pd.concat([test_enc, other_test], axis = 1)
    # convert booleans to integer boolean
    train['complainant_gender'] = (train['complainant_gender'] == 'Male') * 1
    test['complainant_gender'] = (test['complainant_gender'] == 'Male') * 1
    train['substantiated'] = train['substantiated'] * 1
    test['substantiated'] = test['substantiated'] * 1
    display(train)
    display(test)
    
    # Create a BinaryLabelDataset object from the train and test data
    #display(pd.DataFrame(train['substantiated']).head())
    train_bld = BinaryLabelDataset(df= train,
                                   label_names=['substantiated'], protected_attribute_names=['complainant_gender'],
                                   favorable_label=1, unfavorable_label=0,
                                   privileged_protected_attributes=[1])
    test_bld = BinaryLabelDataset(df= test,
                                  label_names=['substantiated'], protected_attribute_names=['complainant_gender'],
                                  favorable_label=1, unfavorable_label=0,
                                  privileged_protected_attributes=[1])

    # Create the debiasing object
    sess = tf.Session()
    debiased_model = AdversarialDebiasing(privileged_groups=[{'complainant_gender': 1}],
                                          unprivileged_groups=[{'complainant_gender': 0}],
                                          scope_name='debiased_classifier',
                                          seed=0, sess = sess)
    #tf.get_variable_scope().reuse_variables()

    # Train the debiased model
    debiased_model.fit(train_bld)

    # Make predictions on the test data
    y_pred = debiased_model.predict(test_bld).labels

    # Compute accuracy and fairness metrics
    accuracy = accuracy_score(test['substantiated'], y_pred)
    bld_metric = BinaryLabelDatasetMetric(test_bld, unprivileged_groups=[{'complainant_gender': 0}],
                                          privileged_groups=[{'complainant_gender': 1}])
    disparate_impact = bld_metric.disparate_impact()
    statistical_parity_difference = bld_metric.statistical_parity_difference()
    equal_opportunity_difference = bld_metric.equal_opportunity_difference()


# In[ ]:





# In[43]:


cats = ["allegation", "contact_reason"]

# Split data into train and test
    
nypd = data

train = nypd.sample(frac=0.7, random_state=42)
test = nypd.drop(train.index)

model(train, test, cats)


# In[26]:



# In[ ]:





# In[ ]:





# In[19]:


# ### Calculating fairnes notions for No Missingness At All

# In[20]:


train, test = train_test_split(nypd, test_size=0.2)


# In[21]:


#storing fairness notions for no missingness
no_missing_fairness = []
no_missing = model(train, test, cat)
no_missing_fairness.append(no_missing)


# ### Fairness notions for NMAR

# In[22]:


train_nmar, test_nmar = train_test_split(nmar, test_size=0.2)


# In[23]:


nmar_fairness = []
nmar_model = model(train_nmar, test_nmar, cat)
nmar_fairness.append(nmar_model)


# ### Fairness notions for MCAR

# In[24]:


train_mcar, test_mcar = train_test_split(mcar, test_size=0.2)


# In[25]:


mcar_fairness = []
mcar_model = model(train_mcar, test_mcar, cat)
mcar_fairness.append(mcar_model)


# ### Fairness notions for MAR

# In[26]:


train_mar, test_mar = train_test_split(mar, test_size=0.2)


# In[27]:


mar_fairness = []
mar_model = model(train_mar, test_mar, cat)
mar_fairness.append(mar_model)


# In[28]:


#put our fairness statistics into arrays for future usage
par= [no_missing_fairness[0][0],nmar_fairness[0][0],mcar_fairness[0][0],mar_fairness[0][0]]
odds= [no_missing_fairness[0][1],nmar_fairness[0][1],mcar_fairness[0][1],mar_fairness[0][1]]
opp = [no_missing_fairness[0][2],nmar_fairness[0][2],mcar_fairness[0][2],mar_fairness[0][2]]


# ## Visualizing Our Results

# In[29]:


import matplotlib.pyplot as plt


# In[30]:


labels = ['No Missingess,', 'NMAR', 'MCAR', 'MAR']


# In[31]:


plt.figure(figsize = (20, 10))
plt.title('Statistical Parities')
plt.xlabel('Missingness Type')
plt.ylabel('Statistical Parotities')
plt.ylim(min(par) - 0.1, max(par) + 0.1)
plt.plot(labels, par, marker='.', markersize = 20)


# In[32]:


plt.figure(figsize = (20, 10))
plt.title('Equality of Odds')
plt.xlabel('Missingness Type')
plt.ylabel('Equality of Odds')
plt.ylim(min(odds)-0.1,max(odds) + 0.1)
plt.plot(labels, odds, marker='.', markersize = 20)


# In[33]:


plt.figure(figsize = (20, 10))
plt.title('Equality of Opportunity')
plt.xlabel('Missingness Type')
plt.ylabel('Equality of Opportunity')
plt.ylim(min(opp)-0.1,max(opp) + 0.1)
plt.plot(labels, opp, marker='.', markersize = 20)


# In[ ]:





# In[ ]:




