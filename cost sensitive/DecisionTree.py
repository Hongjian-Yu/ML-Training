#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[7]:


class D_Tree():
    def __init__(self,train_set):
        if not isinstance(train_set,pd.DataFrame):
            raise ValueError('train_set must be a DataFrame object')
        self.train_set = train_set
        self.train_x = train_set.iloc[:,:-1].values
        self.train_y = train_set.iloc[:,-1].values
    def fit(self):
        clf = DecisionTreeClassifier(criterion='entropy',random_state=42)
        clf.fit(self.train_x,self.train_y)
        self.Tree = clf
    def predict(self,test_x):
        return self.Tree.predict(test_x)


# In[8]:


# tree = D_Tree(pd.DataFrame({'A':[1,2,3,4],'B':[2,3,4,5],'Y':[0,0,1,1]}))
# print(tree.train_x)
# print(tree.train_y)
# tree.fit()
# print(tree.predict([[3,5]]))


# In[ ]:




