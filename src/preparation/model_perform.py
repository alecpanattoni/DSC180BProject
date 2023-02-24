#!/usr/bin/env python
# coding: utf-8

# In[1]:


import aif360
from aif360.sklearn.metrics import statistical_parity_difference
from aif360.sklearn.metrics import average_odds_difference
from aif360.sklearn.metrics import equal_opportunity_difference
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing

import pandas as pd

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
    y_actual = test['substantiated']
    
    # Create a BinaryLabelDataset object from the train and test data
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
                                          sess = sess)

    # Train the debiased model
    debiased_model.fit(train_bld)

    # Make predictions on the test data
    y_pred = debiased_model.predict(test_bld).labels
    y_pred = pd.Series([int(y_pred[i][0]) for i in range(len(y_pred))])
    
    #tf.get_variable_scope().reuse_variables()
    
    sess.close()
    tf.reset_default_graph()

    # Compute accuracy and fairness metrics
    parity = statistical_parity_difference(y_actual, y_pred)
    odds = average_odds_difference(y_actual, y_pred)
    opportunity = equal_opportunity_difference(y_actual, y_pred)
    
    def accuracy(pred, actual):
        assert len(pred) == len(actual)
        corr = 0
        for i in range(len(pred)):
            if pred[i] == actual[i]:
                corr += 1
        return corr / len(pred)
    
    acc = accuracy(y_pred, y_actual)
    
    print('\n')
    print("accuracy: " + str(acc))
    print("Statistical parity: " + str(parity))
    print("Equality of odds: " + str(odds))
    print("Equality of opportunity: " + str(opportunity))
    
    return [acc, parity, odds, opportunity]



# In[ ]:




