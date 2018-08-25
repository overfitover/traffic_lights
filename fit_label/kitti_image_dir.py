import os
import glob


def read_data(file):
    '''
    @description:
    :param file: 文件名
    :return: 读取的数据
    '''
    f = open(file)
    line = f.readlines()
    return line

# print(read_data('/home/yxk/project/data/training/label_2/000001.txt'))

def iterate_data(data_dir):
    f_rgb = glob.glob(os.path.join(data_dir, 'image_2', '*.png'))
    f_label = glob.glob(os.path.join(data_dir, 'label_2', '*.txt'))
    f_rgb.sort()
    f_label.sort()

    print(f_rgb)
    print(f_label)
    data_tag = [name.split('/')[-1].split('.')[-2] for name in f_rgb]
    print(data_tag)

    assert len(data_tag) != 0, "dataset folder is not correct"
    assert len(data_tag) == len(f_rgb), "dataset folder is not correct"

    f = open('kitti_yolo3.txt', 'a')
    for i in range(len(data_tag)):
        f.writelines(f_rgb[i])
        f.writelines(' ')
        line = read_data(f_label[i])
        f.writelines(line[0])
        f.write('\n')
    f.close()

    return 'read over'


file_dir = '/home/yxk/project/yolo_code/yolo3/keras-yolo3/training/'

iterate_data(file_dir)