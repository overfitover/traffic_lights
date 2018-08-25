import numpy as np
import pandas as pd
import os
import os.path as osp

def file_path(name):
    '''
    @description: 该文件所在目录下
    :param name:
    :return:
    '''
    this_dir = osp.dirname(__file__)
    path = osp.join(this_dir, name)
    return path

def read_data(label_file):
    '''
    @description: 将数据转出（N, 5）格式 xin, yin, xmax, ymax, status
    :param file:
    :return:
    '''
    lights_status = {'go': 0, 'stop': 1, 'warning': 2}
    f = pd.read_csv(label_file)
    lines = f.values
    total_data = []
    data = []

    for line in lines:
        line_data = (line[0].split(';')[:6])
        if line_data[1] in {'go', 'stop', 'warning'}:
            data.append(line_data[0].split('/')[1])
            data.append(int(line_data[2]))
            data.append(int(line_data[3]))
            data.append(int(line_data[4]))
            data.append(int(line_data[5]))
            data.append(lights_status[line_data[1]])
        total_data.append(data)
        data = []
    while [] in total_data:
        total_data.remove([])
    return total_data


def read_all_data(file, image_dir, label_file):
    '''
    description:
    :param file_dir: 文件所在父文件夹
    :return: 读取所有的文件
    '''
    label_data = read_data(label_file)
    image_name = []
    for image in os.listdir(image_dir):
        image_name.append(image_dir + image)   # 每张照片的地址
    labels = []
    for image in image_name:
        img_name = image.split('/')[-1]
        for label in label_data:               # 找到图片对应的各种labels
            if img_name == label[0]:
                labels.append(label)

        if len(labels)>0:
            f = open(file, 'a')
            f.writelines(str(image)+' ')
            i = 0
            for label_one in labels:
                i = i+1
                f.writelines(str(label_one[1])+','+str(label_one[2])+','+str(label_one[3])+','+str(label_one[4])+','+str(label_one[5]))
                if i < len(labels):
                    f.writelines(' ')
            f.write('\n')
            labels = []
    return 'write over'

label_file = file_path('{}/lisa-traffic-light-dataset/Annotations/{}/frameAnnotationsBOX.csv'.format(os.getcwd(), 'daySequence1'))
image_dir = file_path('{}/lisa-traffic-light-dataset/{}/frames/'.format(os.getcwd(), 'daySequence1'))

read_data(label_file)
read_all_data('lisa.txt', image_dir, label_file)