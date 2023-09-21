import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy.io
import scipy.linalg
import os
import sklearn.metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
import numpy as np
import sklearn.metrics
from cvxopt import matrix, solvers
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn.metrics import accuracy_score
import argparse
from sklearn.metrics import mean_squared_error

def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(
                np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, None, gamma)
    return K


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = K @ M @ K.T + self.lamb * np.eye(n_eye), K @ H @ K.T
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = A.T @ K
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new
import numpy as np
from scipy import linalg
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
class CCA:
    def __init__(self):
        self.a = None
        self.b = None

    def train(self, X, Y):
        Nx, cx = X.shape
        Ny, cy = Y.shape

        # 标准化 (N, C)
        X = (X - np.mean(X, 0)) / np.std(X, 0)
        Y = (Y - np.mean(Y, 0)) / np.std(Y, 0)

        # 求三个S
        data = np.concatenate([X, Y], axis = 1)
        cov = np.cov(data, rowvar=False)
        N, C = cov.shape
        Sxx = cov[0:cx, 0:cx]
        Syy = cov[cx:C, cx:C]
        Sxy = cov[0:cx, cx:C]
        Sxx_ = linalg.sqrtm(np.linalg.inv(Sxx))
        Syy_ = linalg.sqrtm(np.linalg.inv(Syy))
        M = Sxx_.T.dot(Sxy.dot(Syy_))
        U, S, V = np.linalg.svd(M, full_matrices=False)
        u = U[:, 0]
        v = V[0, :]
        self.a = Sxx_.dot(u)
        self.b = Syy_.dot(v)

    def predict(self, X, Y):
        X_ = X.dot(self.a)
        Y_ = Y.dot(self.b)
        return X_, Y_

    def cal_corrcoef(self, X, Y):
        X_, Y_ = self.predict(X, Y)
        return np.corrcoef(X_, Y_)[0,1]
if __name__ == "__main__":
    tca = TCA(kernel_type='primal', dim=40, lamb=0.1, gamma=1)
    feanum = 3  # 一共有多少特征
    window = 5  # 时间窗设置
    from sklearn.cross_decomposition import CCA
    df1 = pd.read_csv(r"F:\BaiduNetdiskDownload\python\House Price\train_S1004.csv")
    X_train = df1.iloc[:, :-1]
    src_val = df1.iloc[:, -1]
    df2 = pd.read_csv(r"F:\BaiduNetdiskDownload\python\House Price\train_T1004.csv")
    depth = df2.iloc[:3727, 0].values
    Y_train = df2.iloc[:, :-1]
    tar_val = df2.iloc[:, -1]
    cca_sklearn = CCA(n_components=2)
    cca_sklearn.fit(X_train, Y_train)
    X_c, Y_c = cca_sklearn.transform(X_train, Y_train)
    print(X_c.shape)
    print(src_val.shape)
    # 将src_val转换为二维数组
    src_val_2d = np.column_stack((src_val,))
    # 使用numpy.concatenate函数将X_c和src_val进行横向拼接
    df1 = np.concatenate((X_c, src_val_2d), axis=1)
    # 将tar_val转换为二维数组
    tar_val_2d = np.column_stack((tar_val,))
    # 使用numpy.concatenate函数将X_c和src_val进行横向拼接
    df2 = np.concatenate((Y_c, tar_val_2d), axis=1)
    # 进行归一化操作
    min_max_scaler1 = preprocessing.MinMaxScaler()
    df3 = min_max_scaler1.fit_transform(df1)
    df1 = pd.DataFrame(df1)
    df4 = pd.DataFrame(df3, columns=df1.columns)

    min_max_scaler2 = preprocessing.MinMaxScaler()
    df5 = min_max_scaler2.fit_transform(df2)
    df2 = pd.DataFrame(df2)
    df6 = pd.DataFrame(df5, columns=df2.columns)
    Xs = df4.iloc[:, :-1]
    Ys = df4.iloc[:, -1]
    Xt = df6.iloc[:, :-1]
    Yt = df6.iloc[:, -1]
    Xs_new, Xt_new = tca.fit(Xs, Xt)
    df4.iloc[:, :-1] = Xs_new
    df6.iloc[:, :-1] = Xt_new

    stock1 = df4
    seq_len = window
    amount_of_features = len(stock1.columns)  # 有几列
    data1 = stock1.values  # pd.DataFrame(stock) 表格转化为矩阵
    sequence_length = seq_len + 1  # 序列长度+1
    result1 = []
    for index in range(len(data1) - sequence_length):  # 循环 数据长度-时间窗长度 次
        result1.append(data1[index: index + sequence_length])  # 第i行到i+5
    result1 = np.array(result1)  # 得到样本，样本形式为 window*feanum

    stock = df6
    seq_len = window
    amount_of_features = len(stock.columns)  # 有几列
    data = stock.values  # pd.DataFrame(stock) 表格转化为矩阵
    sequence_length = seq_len + 1  # 序列长度+1
    result = []
    for index in range(len(data) - sequence_length):  # 循环 数据长度-时间窗长度 次
        result.append(data[index: index + sequence_length])  # 第i行到i+5
    result = np.array(result)  # 得到样本，样本形式为 window*feanum
    # print(result)

    # 分训练集测试集 最后cut个样本为测试集
    x_train = result1[:, :-1]
    y_train = result1[:, -1][:, -1]
    x_test = result[:, :-1]
    y_test = result[:, -1][:, -1]
    X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))


    d = 0.01
    model = Sequential()
    model.add(LSTM(32, input_shape=(window, feanum), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(1, input_shape=(window, feanum), return_sequences=False))
    model.add(Dropout(d))
    model.compile(loss='mse', optimizer='adam')

    model.fit(X_train, y_train, epochs=50, batch_size=256)  # 训练模型nb_epoch次
    # 导入包
    import joblib

    # 保存模型
    joblib.dump(model, 'model.dat')  

    # 加载模型
    loaded_model2 = joblib.load('model.dat')
    y_train_predict = model.predict(X_train)[:, 0]
    df11 = df1.iloc[:-6]
    y_train = df11.iloc[:, -1]
    df9 = df4.iloc[:-6]
    df9.iloc[:, -1] = y_train_predict.flatten()

    y_test_predict = model.predict(X_test)[:, 0]
    df12 = df2.iloc[:-6]
    y_test = df12.iloc[:, -1]
    df10 = df6.iloc[:-6]
    df10.iloc[:, -1] = y_test_predict.flatten()

    # 反归一化
    df91 = min_max_scaler1.inverse_transform(df9)
    # 反归一化
    df101 = min_max_scaler2.inverse_transform(df10)

    y_train_predict = df91[:, -1]
    y_test_predict = df101[:, -1]

    plt.figure(figsize=(3, 10))
    plt.plot(y_test, depth, label='True', color='blue', linestyle='-', linewidth=2)
    plt.plot(y_test_predict, depth, label='Predicted', color='red', linestyle='-', linewidth=2)
    #plt.scatter([5, 10, 15], [20, 40, 60], color='green', marker='o', label='Points')  # Add points
    plt.title('CCA-TCA-LSTM', fontsize=16)
    plt.xlabel('pp/(g/cm³)', fontsize=14)
    plt.ylabel('depth', fontsize=14)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # Calculate metrics for evaluation
    mse = np.mean((y_test_predict - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_predict - y_test))

    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)

    '''y_test_predict = model.predict(X_test)[:, 0]
    y_test = y_test'''
    # 使用模型
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    import math

    def mape(y_true, y_pred):
        return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


    print('训练集上的MAE/MSE/MAPE')
    print(mean_absolute_error(y_train_predict, y_train))
    print(mean_squared_error(y_train_predict, y_train))
    print(mape(y_train_predict, y_train))
    print('测试集上的MAE/MSE/MAPE')
    print(mean_absolute_error(y_test_predict, y_test))
    print(mean_squared_error(y_test_predict, y_test))
    print(mape(y_test_predict, y_test))