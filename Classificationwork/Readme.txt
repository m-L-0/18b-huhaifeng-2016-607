������������
ѧ�ţ�2016011607
1.���빤��
import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.io import loadmat,savematfrom sklearn.decomposition import PCA
2.��������
3.����PCA�����ݽ��н�άԤ����
pca = PCA(n_components=40)
D=pca.fit_transform(data)
T=pca.fit_transform(test)

4.�������ɭ���㷨���з���

