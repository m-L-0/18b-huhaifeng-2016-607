姓名：胡海锋
学号：2016011607
1.导入工具
import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.io import loadmat,savematfrom sklearn.decomposition import PCA
2.导入数据
3.利用PCA对数据进行降维预处理
pca = PCA(n_components=40)
D=pca.fit_transform(data)
T=pca.fit_transform(test)

4.利用随机森林算法进行分类

