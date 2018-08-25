import numpy as np
import os
import os.path as osp

def file_path(name):
    this_dir = osp.dirname(__file__)
    path = osp.join(this_dir, name)
    return path

lights_status = {'go': 0, 'stop': 1, 'warning':2, 'ambiguous':3}
def read_data(file):
    '''
    @description: 将数据转出（N, 5）格式 xin, yin, xmax, ymax, status
    :param file:
    :return:
    '''
    f = open(file)
    lines = f.readlines()
    total_data = []
    data = []
    for line in lines:
        data_line = list(line.split())
        data.append(int(data_line[2]))
        data.append(int(data_line[3]))
        data.append(int(data_line[4]))
        data.append(int(data_line[5]))
        data.append(int(data_line[6]))
        status = list(data_line[-1].split('\''))
        if status[1] in {'go', 'stop', 'warning', 'ambiguous'}:
            data.append(lights_status[status[1]])
        total_data.append(data)
        data = []
    return total_data

file = './label.txt'
print(read_data(file))



def write_data(file, data):
    f = open(file, 'a')
    for i in range(len(data)):
        for j in range(5):                 # 一个object数据
            f.writelines(str(data[i][j]))
            if j < 4:
                f.writelines(',')
        if i < len(data)-1:
            f.writelines(' ')
    f.close()


def read_all_data(image_dir, label_dir):
    '''
    description:
    :param file_dir: 文件所在父文件夹
    :return: 读取所有的文件
    '''
    label_data = read_data(label_dir)
    image_name = []
    for image in os.listdir(image_dir):
        image_name.append(image_dir + image)   # 每张照片的地址
    print(image_name)
        #write_data(file_name, data)

    return 'write over'


image_dir = file_path('image')
label_dir = file_path('label.txt')
# read_all_data(image_dir, label_dir)