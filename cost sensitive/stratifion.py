#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


# In[12]:


def stratified_sampling(dataFrame, cost_matrix, method='undersampling', random_state=None):
    """
    对数据集进行分层抽样，用于处理成本敏感学习问题。

    参数:
    X: 特征矩阵，形状为 (n_samples, n_features)。
    y: 标签向量，形状为 (n_samples,)。
    cost_matrix: 成本矩阵，形状为 (n_classes, n_classes)。
    method: 抽样方法，可以是 'undersampling' 或 'oversampling'。
    random_state: 随机种子，用于控制随机数生成。

    返回值:
    X_resampled: 重新采样后的特征矩阵。
    y_resampled: 重新采样后的标签向量。
    """
    X = dataFrame.iloc[:,:-1].values
    y = dataFrame.iloc[:,-1].values

    y = np.array(y)
    X = np.array(X)
    
    classes,counts = np.unique(y,return_counts=True)

    class_prob = counts / np.sum(counts)

    class_bl = counts / np.min(counts)

    

    n_classes = len(classes)

    # 计算类别权重
    class_cost_sum = np.sum(cost_matrix, axis=0)

    
    sample_p = class_prob * class_cost_sum
    
    
    sample_prob = sample_p / np.sum(sample_p)

    sample_bl = sample_prob / np.min(sample_prob)

    
    
    # sample_num = np.sum(counts) * sample_prob
    
    # sample_num = [int(i) for i in sample_num]

    bili = [sample_prob[i]/class_prob[i] for i in range(len(sample_bl))]

    
    
    
    if method == 'undersampling':
        bili_index = np.argmax(bili)
        bili_num = counts[bili_index]
        # sample_index = np.where(sample_num<counts)
        sample_index = [i for i in range(len(classes)) if i!= bili_index]
        
        for i in sample_index:
            
            

            sizes  = counts[i] - bili_num / sample_bl[bili_index] * sample_bl[i]

            for j in range(int(sizes)):
            
                x_index = np.where(y == classes[i])
                
                index_to_delete = np.random.choice(x_index[0],size=1,replace=False)

                X = np.delete(X,index_to_delete,axis=0)
                y = np.delete(y,index_to_delete)

    if method == 'oversampling':
        bili_index = np.argmin(bili)

        bili_num = counts[bili_index]
        # sample_index = np.where(sample_num>counts)
        sample_index = [i for i in range(len(classes)) if i!= bili_index]
       
        for i in sample_index:

            

            sizes  =  bili_num / sample_bl[bili_index] * sample_bl[i] - counts[i]

            for j in range(int(sizes)):

                x_index = np.where(y == classes[i])
                index_to_add = np.random.choice(x_index[0],size=1,replace=False)

                X = np.concatenate([X, X[index_to_add]])
                y = np.concatenate([y, [classes[i]]])
 

    return X,y


# In[ ]:





# In[ ]:




