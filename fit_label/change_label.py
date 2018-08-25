import numpy as np
import os

dict = {'Car': 0 , 'Van': 1, 'Truck': 2, 'Pedestrian': 3, 'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7}

def read_data(file):
    '''
    @description:读取原始数据并转换成需要数据
    :param file: 文件名
    :return: 读取的数据
    '''
    f = open(file)
    lines = f.readlines()
    data = []
    line_data = []
    for line in lines:
        data_line = list(line.split())
        if data_line[0] in {'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc'}:
            line_data.append(int(float(data_line[-11])))
            line_data.append(int(float(data_line[-10])))
            line_data.append(int(float(data_line[-9])))
            line_data.append(int(float(data_line[-8])))
            line_data.append(dict[data_line[0]])
            data.append(line_data)
            line_data = []
    return data

# print(read_data('/home/yxk/project/data/training/label_2/000001.txt'))

def write_data(file, data):
    f = open(file, 'a')
    for i in range(len(data)):             #
        for j in range(5):                 # 一个object数据
            f.writelines(str(data[i][j]))
            if j < 4:
                f.writelines(',')
        if i < len(data)-1:
            f.writelines(' ')
    f.close()


def read_all_data(file_dir):
    '''
    description:
    :param file_dir: 文件所在父文件夹
    :return: 读取所有的文件
    '''
    for file in os.listdir(file_dir):
        file_name = file_dir + file
        data = read_data(file_name)
        #print(data)
        if os.path.exists(file_name):
            os.remove(file_name)
        write_data(file_name, data)

    return 'write over'

file_dir = '/home/yxk/project/yolo_code/yolo3/keras-yolo3/training/label_2/'
read_all_data(file_dir)

