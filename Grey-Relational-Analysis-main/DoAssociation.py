from GRA_model import GRA_Model
import pandas as pd
import numpy as np
import time


class Association:
    def __init__(self, matrix):
        self.matrix = matrix
        self.pairs = set()
        self.failed_dict = {}

    def __find_max_index_once(self):
        # TODO 这可能会出现一个问题,即最大值同时可能有几个,后面再解决这个问题,现在只假设有一个
        max_element = np.max(self.matrix)
        index_tuple = np.where(self.matrix == max_element)

        # 因为np.where()返回的是Tuple:(np.array, np.array),两个array对象分别是是行和列坐标信息
        # camera为行,radar为列,例如(0，5)表示第0行camera轨迹 与 第5行radar轨迹相关
        index_tuple = (index_tuple[0][0], index_tuple[1][0])
        return max_element, index_tuple

    def __find_max_index_all(self):
        # 只要有一个不为0,就返回True
        while np.any(self.matrix):
            _, index_tuple = self.__find_max_index_once()

            self.pairs.add(index_tuple)
            self.__set_zero(index_tuple)
            # index_tuple = self.__set_zero(index_tuple)

        # 在确定关联矩阵self.matrix全为0之后,就要检查哪些是关联失败的
        # self.failed_dict = {'camera': set(), 'radar': set()}
        failed_camera_set = {i for i in range(self.matrix.shape[0])} - {j[0] for j in self.pairs}
        failed_radar_set = {i for i in range(self.matrix.shape[1])} - {j[1] for j in self.pairs}

        self.failed_dict = {'camera': failed_camera_set, 'radar': failed_radar_set}

    def __set_zero(self, index):
        # 将输入的matrix中的index坐标信息的行列全置为0
        self.matrix[index[0], :] = 0
        self.matrix[:, index[1]] = 0
        return self.matrix

    def run(self):
        self.__find_max_index_all()

    def get_pairs_and_failed(self):
        # 注意这里返回时又嵌套了一层tuple
        return self.pairs, self.failed_dict


if __name__ == "__main__":
    t1 = time.time()
    # cors_dict = {}
    # failed_dict = {'camera': None, 'radar': None}
    # cam_data = pd.read_csv('01.csv')
    # radar_data = pd.read_csv('02.csv')
    # model = GRA_Model(cam_data, radar_data)
    # model.run()
    # matrix = model.get_matrix()
    # print("关联度矩阵:", matrix)

    a = np.array([[1, 1, 1],
                  [2, 2, 2],
                  [3, 3, 3],
                  [4, 9, 4]])

    asso = Association(a)
    asso.run()
    print(asso.get_pairs_and_failed())

    t2 = time.time() - t1
    print("RUN TIME: {} ".format(t2))
