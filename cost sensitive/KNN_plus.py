#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# from sklearn.metrics.pairwise import euclidean_distances
# from scipy.spatial.distance import hamming


# In[2]:


def custom_distance(x1,x2,feature_label,euclidean,ham):
    
    euclidean_features = euclidean
    hamming_features = ham
    # euclidean_features = [i for i in range(0,Wine_data_.shape[1]-1)]
    # hamming_features=[]
    distance = 0
    flag=0
    if ham is None:
        for i in range(len(feature_label)):
            distance+=(x1[i]-x2[i])**2
            flag=1
    elif euclidean is None:
        for i in range(len(feature_label)):
            distance+=(x1[i]!=x2[i])
    else:
        for i in range(len(feature_label)):
            if feature_label[i] in euclidean_features:
                distance+=(x1[i]-x2[i])**2
                flag=1
            else:
                distance+=(x1[i]!=x2[i])
    
    if flag==1:
        distance = np.sqrt(distance)
    # euclidean_dist = euclidean_distances([[x1[i] for i in euclidean_features]], [[x2[i] for i in euclidean_features]])[0][0]
    return distance


# In[3]:


class KNN(object):
    def __init__(self,feature_subset_size,train_set,resample_times):
        if not isinstance(train_set,pd.DataFrame):
            raise ValueError('train_set must be a DataFrame object')
        new_index = list(range(len(train_set)))
        train_set.index = new_index
        colums = [i for i in range(train_set.shape[1]-1)]
        self.feature_subset_size = feature_subset_size
        self.train_set = train_set[colums]
        self.resample_times = resample_times
        # self.model = model
        self.train_label = train_set.iloc[:,-1].values
    
    
    def fit(self,C,n_neighbor,euclidean_features=None,hamming_features=None):
        self.C = C
        all_colums_list = self.train_set.columns.tolist()
        # remain_colums = [colums_name for colums_name in all_colums_list if colums_name!='target']
        S=[]
        F=[]
        for i in range(self.resample_times):
            select_colums = np.random.choice(all_colums_list,self.feature_subset_size,replace=False)
            # print(select_colums)
            select_ = self.train_set[select_colums].copy()
            X = select_.values
            # select_['target'] = self.train_label
            # print(select_colums)
            knn =KNeighborsClassifier(n_neighbors=n_neighbor,metric=custom_distance,metric_params={'feature_label':select_colums,'euclidean':euclidean_features,'ham':hamming_features})
            
            # new_model = clone(knn)
            # knn.effective_metric_params_ = {}
                                            
            
            # knn =KNeighborsClassifier(n_neighbors=2,metric=custom_distance,metric_params={})
            # print("X:",X.shape)
            # Y = select_['target'].values
            Y = self.train_label
            S.append(knn.fit(X,Y))
            F.append(select_colums)
        relabel = []
        
        for i in range(len(self.train_set)):
            P_each_model = np.array([S[j].predict([self.train_set[F[j]].copy().values[i]]) for j in range(len(S))]).flatten()
            # print(P_each_model)
            unique,counts = np.unique(P_each_model,return_counts=True)
            
            # print(counts)
            P = np.array(counts*1.0 / len(P_each_model))
            all_class = np.unique(self.train_label,return_counts=False)
            class_cate = {i:0 for i in all_class}
            for element, proportion in zip(unique,P):
                class_cate[element] = proportion
            # print(P)
            P_list = np.array([class_cate[j] for j in all_class]) 
           
            relabel.append(np.argmin(self.C.dot(P_list.T)))
            
        self.relabel = relabel    
        
        for i in range(len(S)):
            S[i].fit(self.train_set[F[i]].copy().values,relabel)    
        
        self.S = S
        self.F = F
            
    def predict(self,test_set):
        F = self.F
        S = self.S
        # colums = [i for i in range(test_set.shape[1])-1]
        self.test_set = test_set.iloc[:,:-1]
        self.test_label = test_set.iloc[:,-1].values
        label=[]
        for i in range(self.resample_times):
            sample = self.test_set[F[i]].values
            label.append(S[i].predict(sample))
        label_ = np.array(label).T
        true_label = []
        for i in label_:
            unique, counts = np.unique(i, return_counts=True)
            most_number = unique[np.argmax(counts)]
            true_label.append(most_number)
       
    
       
        return true_label

        


# In[4]:


# def make_cost_matrix(label):
#     # np.random.seed(0)
#     label_list,counts = np.unique(label,return_counts=True)
#     class_number = len(label_list)
#     data = np.zeros((class_number,class_number))
#     Data = pd.DataFrame(data,columns=label_list,index=label_list)
#     for name1 in label_list:
#         for name2 in label_list:
#             if name1 == name2:
#                 Data[name1][name2] = np.random.randint(0,1000)
#             else:
#                 Data[name1][name2] = np.random.randint(0,2000*counts[np.where(label_list==name1)] / counts[np.where(label_list==name2)])
#                 # Data[name1][name2] = np.random.uniform(0,10000)
#     return Data
# def compute_cs(cost_matrix,true_label,predict_label):
#     cs = 0

#     for i in range(len(true_label)):
        
#         # if true_label[i]!=predict_label[i]:
#         cs += cost_matrix.iloc[predict_label[i]][true_label[i]]

#     # conf_matrix = confusion_matrix(true_label, predict_label)
#     # total_cost = np.sum(conf_matrix * cost_matrix.values)

#     # return total_cost/len(true_label)
#     return cs/len(true_label)
    


# In[5]:


# iris = load_iris()
# iris_data = iris.data
# iris_label = iris.target
# iris_ = pd.DataFrame(iris_data)
# column = iris_.columns
# iris_ = (iris_[column] - iris_[column].min()) / (iris_[column].max() - iris_[column].min())
# iris_['Y'] = iris_label
# cs = []
# cs_k = []

# cost_matrix = make_cost_matrix(iris_label)
# iris_train_set,iris_test_set = train_test_split(iris_,test_size=0.3,random_state=42)
# iris_test_data = iris_test_set.iloc[:,:-1].values
# iris_test_label = iris_test_set.iloc[:,-1].values
# knn = KNN(2,iris_train_set,10)
# knn.fit(cost_matrix.values,euclidean_features=[0,1,2,3])
# predict_label = knn.predict(iris_test_set)
# print(compute_cs(cost_matrix,iris_test_label,predict_label))


# In[6]:


# def make_cost_matrix(train_set):
#     label_set = train_set['target'].values
#     unique,counts = np.unique(label_set,return_counts=True)
#     # print(unique)
#     # print(counts)
#     cost_matrix = np.zeros((len(unique),len(unique)),dtype=int)
#     for i in range(len(unique)):
#         for j in range(len(unique)):
#             if i==j:
#                 cost_matrix[i][j] = 0
#             else:
#                 # print(unique[i]," ",unique[j])
#                 cost_matrix[i][j] = 2000*counts[unique[i]] / counts[unique[j]]
#     return cost_matrix


# In[7]:


# iris = load_iris()
# iris_feature = iris.data
# iris_class = iris.target
# iris_DataFrame_feature= pd.DataFrame(iris_feature)
# iris_DataFrame = iris_DataFrame_feature.copy()
# iris_DataFrame['target'] = iris_class
# train_DataFrame = iris_DataFrame


# In[8]:


# Wine_data = pd.read_csv('data_set/wine.data',header=None)
# Wine_data.columns = ['target'] + [i for i in range(0,Wine_data.shape[1]-1)]
# unique = np.unique(Wine_data['target'].values,return_counts=False)
# replace_dict = {unique[i]:i for i in range(len(unique))}
# # print(replace_dict)
# Wine_data['target'].replace(replace_dict,inplace=True)
# print(Wine_data)
# Wine_data_ = (Wine_data.iloc[:,1:]-Wine_data.iloc[:,1:].min())/(Wine_data.iloc[:,1:].max()-Wine_data.iloc[:,1:].min())
# Wine_data_['target'] = Wine_data['target'].values
# # print(data)
# print(Wine_data_)
# train_DataFrame = Wine_data_


# In[9]:


# Anneal_data = pd.read_csv('data_set/anneal.data',header=None)
# Anneal_data.columns = [i for i in range(0,Anneal_data.shape[1]-1)] + ['target']
# # print(Anneal_data)
# unique = np.unique(Anneal_data['target'].values,return_counts=False)
# replace_dict = {unique[i]:i for i in range(len(unique))}
# Anneal_data['target'].replace(replace_dict,inplace=True)

# continue_columns = [3,4,8,32,33,34]
# scatter_columns = [i for i in range(0,Anneal_data.shape[1]-1) if i not in continue_columns]
# for i in scatter_columns:
#     unique_ = np.unique(Anneal_data[i].values,return_counts=False)
#     replace_ = {unique_[i]:i for i in range(len(unique_))}
#     Anneal_data[i].replace(replace_,inplace=True)
# train_DataFrame = Anneal_data


# # C = np.array([[0, 1000, 1500], [2810, 0, 2292], [11, 16, 0]])
# # print(iris_DataFrame)
# sums=[]
# cs_sum=[]
# for i in range(1,37,3):
#     train_set,test_set = train_test_split(train_DataFrame,test_size=0.3)
#     C = make_cost_matrix(train_set)
    
#     knn_classifier = KNN(i,train_set,100)
#     knn_classifier.fit(C)
#     acc,cs = knn_classifier.predict(test_set)
#     sums.append(acc)
#     cs_sum.append(cs/len(test_set))
#     print("acc:",acc,"   cs:",cs/len(test_set))
   
# print("average:",sum(sums)*1.0/len(sums))
# print("average_cs:",sum(cs_sum)*1.0/len(cs_sum))
# # print(type(sum))


# In[10]:


# # C = np.array([[0, 1000, 1500], [2810, 0, 2292], [11, 16, 0]])
# # print(iris_DataFrame)
# sums=[]
# cs_sum=[]
# for i in range(20):
#     train_set,test_set = train_test_split(train_DataFrame,test_size=0.3)
#     C = make_cost_matrix(train_set)
    
#     knn_classifier = KNN(6,train_set,50)
#     knn_classifier.fit(C,3)
#     acc,cs = knn_classifier.predict(test_set)
#     sums.append(acc)
#     cs_sum.append(cs/len(test_set))
#     print("acc:",acc,"   cs:",cs/len(test_set))
   
# print("average:",sum(sums)*1.0/len(sums))
# print("average_cs:",sum(cs_sum)*1.0/len(cs_sum))
# # print(type(sum))


# In[ ]:





# In[ ]:




