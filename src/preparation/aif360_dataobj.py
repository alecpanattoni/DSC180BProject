#!/usr/bin/env python
# coding: utf-8

# In[1]:


import aif360


# In[2]:


def make_ai360_object(df, pred, protected, privileged, categorical, favorable_func):
    """
    creates an ai360 dataframe object. The parameters are the following:
    df: pandas dataframe object
    pred: str of label outcome column 
    protected: list of columns of protected attributes
    privileged: list of lists, where each internal list contains the value(s) considered "privileged" for each
                of the listed protected attributes
    categorical: list of categorical columns as strings 
    favorable_func: simple lambda function entailing which values of pred indicate discrimination
    """
    df_obj = aif360.datasets.StandardDataset(
        df,
        label_name = pred,
        favorable_classes = favorable_func,
        protected_attribute_names = protected,
        privileged_classes = privileged,
        categorical_features = categorical
    )
    return df_obj

