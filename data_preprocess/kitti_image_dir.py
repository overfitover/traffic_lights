import os
import glob


def read_data(file):
    '''
    @description:
    :param file: 文件名
    :return: 读取的数据
    '''
    f = open(file)
    lines = f.readlines()
    return lines


def iterate_data(data_dir):
    f_rgb = glob.glob(os.path.join(data_dir, 'images', '*.jpg'))
    f_label = glob.glob(os.path.join(data_dir, 'labels', '*.txt'))
    f_rgb.sort()
    f_label.sort()

    data_tag = [name.split('/')[-1].split('.')[-2] for name in f_rgb]

    assert len(data_tag) != 0, "dataset folder is not correct"
    assert len(data_tag) == len(f_rgb) , "dataset folder is not correct"

    f = open('apollo_data.txt', 'a')
    for i in range(len(data_tag)):
        f.writelines(f_rgb[i])
        f.writelines(' ')
        line = read_data(f_label[i])
        f.writelines(line[0])
        f.write('\n')
    f.close()

    return 'read over'


file_dir = '/home/yxk/project/traffic_light/keras-yolo3/data_preprocess/trainsets/'

iterate_data(file_dir)