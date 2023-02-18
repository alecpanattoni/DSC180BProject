#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from aif360.algorithms.inprocessing import AdversarialDebiasing
import numpy as np
import pandas as pd
from aif360.sklearn.metrics import statistical_parity_difference
from aif360.sklearn.metrics import average_odds_difference
from aif360.sklearn.metrics import equal_opportunity_difference
from src.preparation import aif360_dataobj

# In[6]:


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
    y_test = test.substantiated.values.astype('int')
    
    pred = mod.predict(test_feats)
    
    parity = statistical_parity_difference(pd.Series(y_test), pd.Series(pred))
    odds = average_odds_difference(pd.Series(y_test), pd.Series(pred))
    opportunity = equal_opportunity_difference(pd.Series(y_test), pd.Series(pred))
    
    def accuracy(pred, actual):
        assert len(pred) == len(actual)
        corr = 0
        for i in range(len(pred)):
            if pred[i] == actual[i]:
                corr += 1
        return corr / len(pred)
    
    print("accuracy: " + str(accuracy(pred, y_test)))
    print("atatistical parity: " + str(parity))
    print("equality of odds: " + str(odds))
    print("equality of opportunity: " + str(opportunity))
    
    
    return [accuracy(pred, y_test),parity,odds,opportunity]

# In[ ]:


#model for finding fairness notions when NaN values are present in the data
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
    y_test = test.substantiated.values.astype('int')
    
    pred = mod.predict(test_feats)
    
    parity = statistical_parity_difference(pd.Series(y_test), pd.Series(pred))
    odds = average_odds_difference(pd.Series(y_test), pd.Series(pred))
    opportunity = equal_opportunity_difference(pd.Series(y_test), pd.Series(pred))
    
    def accuracy(pred, actual):
        assert len(pred) == len(actual)
        corr = 0
        for i in range(len(pred)):
            if pred[i] == actual[i]:
                corr += 1
        return corr / len(pred)
    
    print("accuracy: " + str(accuracy(pred, y_test)))
    print("statistical parity: " + str(parity))
    print("equality of odds: " + str(odds))
    print("equality of opportunity: " + str(opportunity))
    
    
    return [accuracy(pred, y_test),parity,odds,opportunity]
