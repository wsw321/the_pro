# -*- coding: utf-8 -*-
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os


class GRA_Model:
    # Grey Relation Analysis Model For Camera and Radar
    def __init__(self, cam_data, radar_data, p=0.5):
        """
        :param inputData: ndarray
        :param p: usually 0.5
        """
        self.cam_data = np.array(cam_data)
        self.radar_data = np.array(radar_data)
        self.p = p
        
    def __standarOpt(self):
        # # fit()根据input_data的形状创建转换器
        # scaler = StandardScaler().fit(self.input_data)
        # # trasform()用于转换
        # self.input_data = scaler.transform(self.input_data)

        # 另一种预处理方式:除以均值
        cam_mean_all = self.cam_data.sum(axis=0) / self.cam_data.shape[0]
        radar_mean_all = self.radar_data.sum(axis=0) / self.radar_data.shape[0]
        self.cam_data = self.cam_data / cam_mean_all
        self.radar_data = self.radar_data / radar_mean_all

    def __computing_single(self, input_data):
        # The first column is the parent column, and the absolute difference with other columns is obtained
        momCol = input_data[:, 0]
        sonCol = input_data[:, 1:]

        # 计算|X0 - Xi|,Xi为第i个子序列,结果为一维向量
        # 其结果仍返回给sonCol对象(time * 子序列个数)
        for col in range(sonCol.shape[1]):
            sonCol[:, col] = abs(sonCol[:, col]-momCol)

        # 找到子序列中最大最小值,最大值作为系数b,最小值作为系数a
        minMin = sonCol.min()  # a
        maxMax = sonCol.max()  # b

        # 计算关联系数矩阵,sonCol为矩阵(time * 子序列个数),则cors为同类型矩阵
        cors = (minMin + self.p*maxMax)/(sonCol+self.p*maxMax)

        # 找到我们需要的关联度,对时间维度进行求均值,即meanCors为1*子序列个数或子序列个数*1
        meanCors = cors.mean(axis=0)

        return meanCors

    def __computing_all(self):

        result_matrix = np.array([])
        for i in range(self.cam_data.shape[1]):
            new_matrix = np.concatenate((self.cam_data[:, i].reshape(-1, 1), self.radar_data), axis=1)
            mean_cor_temp = self.__computing_single(new_matrix)
            result_matrix = np.append(result_matrix, mean_cor_temp)

        self.result_matrix = result_matrix.reshape(self.cam_data.shape[1], -1)



    def run(self):
        self.__standarOpt()
        self.__computing_all()

    def get_cors(self):
        return self.result['cors']
    
    def get_matrix(self):
        return self.result_matrix


if __name__ == "__main__":
    cam_data = pd.read_csv('01.csv')
    radar_data = pd.read_csv('02.csv')
    print(cam_data)
    print(radar_data)
    model = GRA_Model(cam_data, radar_data)
    model.run()
    print('----------------')
    matrix = model.get_matrix()
    print(matrix)
