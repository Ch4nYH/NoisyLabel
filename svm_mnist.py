#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np # 
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
import svmutil

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')
sns.set_style("whitegrid")


# In[2]:


def load_data():
    iris_data = fetch_openml("mnist_784") # 读取数据
    x = iris_data.data
    y = iris_data.target
    print("Data Shape: ", x.shape) # 数据形状
    print("Label Shape: ", y.shape) # 标签形状
    return x,y # 返回

def train(x, y, kernel, search_params, **options):
    print("Params to search: ", search_params.keys()) # 打印需要搜索的参数
    svc = SVC(kernel= kernel) # 定义基础分类器
    grid = GridSearchCV(svc, search_params, **options) # 使用交叉验证搜索超参
    grid.fit(x, y)
    return grid
    


# In[3]:


def visualize(grid, param):
    mean_test_score = grid.cv_results_['mean_test_score'] # 获得平均结果
    param_C = grid.cv_results_['param_' + param] # 获取搜索的参数
    sns.lineplot(param_C, mean_test_score)
    plt.show()


# # 尝试线性核SVM

# In[4]:


x, y = load_data()
search_params = { 
    'C': np.linspace(0, 1, 100) # 定义超参的搜索范围
}

grid = train(x, y, 'linear', search_params, cv = 5, n_jobs = -1) # 训练


# In[5]:


visualize(grid, 'C')


# In[6]:


def evaluate(grid, x, y):
    acc = grid.score(x, y) # 计算正确率
    print("Accuracy: ", acc) # 打印
    y_pred = grid.predict(x) # 获得预测的标签
    cm = confusion_matrix(y, y_pred) # 画出混淆矩阵
    sns.heatmap(cm, annot = True)
    plt.show()


# In[7]:


evaluate(grid, x, y)