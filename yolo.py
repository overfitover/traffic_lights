#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image

class YOLO(object):
    def __init__(self):
        self.model_path = 'model_data/train_185.h5'  # model path or trained weights path
        self.anchors_path = 'model_data/traffic_lights_anchors.txt'
        self.classes_path = 'model_data/lisa_classes.txt'
        self.score = 0.3
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

        self.num = 0

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (
                None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })  # (x0, y0, x1, y1) [置信度] [类别序号]

        self.num = len(out_boxes)
        # print('out_boxes', out_boxes, 'out_scores',out_scores,'out_classes', out_classes)
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # print(self.num)
        # print('out_boxes:', out_boxes, '\n', 'out_scores: ',out_scores, '\n', 'out_classes', out_classes)

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))  # 一种字体
        thickness = (image.size[0] + image.size[1]) // 300                      # 框的厚度

        a_label = []
        a_score = []
        a_top = []
        a_left = []
        a_bottom = []
        a_right = []

        for i, c in reversed(list(enumerate(out_classes))):   # 倒序
            predicted_class = self.class_names[c]             # 类别名称
            print('predicted_class', predicted_class)
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)  # 字符
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)    # 字符大小


            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            a_label.append(predicted_class)
            a_score.append(score)
            a_top.append(top)
            a_left.append(left)
            a_bottom.append(bottom)
            a_right.append(right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):   # object 框
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])  # 类别不同颜色不同
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])                                   # 字符背景
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)   # 字符
            del draw

        self.a_label = a_label
        self.a_score = a_score
        self.a_top = a_top
        self.a_left = a_left
        self.a_bottom = a_bottom
        self.a_right = a_right

        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()


def _get_class():
    classes_path = 'model_data/coco_classes.txt'
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def draw_label(image, label):
    class_names = _get_class()
    # (x0, y0, x1, y1) [置信度] [类别序号]
    out_boxes = [[100, 200, 300, 400]]
    out_scores = [1]
    out_classes = [4]

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))  # 一种字体
    thickness = (image.size[0] + image.size[1]) // 300                      # 框的厚度
    for i, c in reversed(list(enumerate(out_classes))):   # 倒序
        predicted_class = class_names[c]      # 类别名称
        print('predicted_class', predicted_class)
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)  # 字符
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)    # 字符大小
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        hsv_tuples = [(x / len(class_names), 1., 1.)
                      for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):   # object 框
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])  # 类别不同颜色不同
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])                                   # 字符背景
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)   # 字符
        image.show()
        del draw
    return image

def detect_img_for_map(yolo):
    import glob
    file_dir = '/home/yxk/project/traffic_light/keras-yolo3/data_preprocess/trainsets/'
    f_rgb = glob.glob(os.path.join(file_dir, 'images', '*.jpg'))
    f_rgb.sort()

    for i in range(len(f_rgb)):
        print(f_rgb[i])
        data_tag = [f_rgb[i].split('/')[-1].split('.')[-2]]
        f = open('./map_dir/'+str(data_tag[0])+'.txt', 'a')     # 打开的文件不同,最后就会写入哪个文件
        image = Image.open(f_rgb[i])
        r_image = yolo.detect_image(image)

        for i in range(yolo.num):
            f.writelines(str(yolo.a_label[i]))
            f.writelines(' ')
            f.writelines(str(yolo.a_score[i]))
            f.writelines(' ')
            f.writelines(str(yolo.a_left[i]))
            f.writelines(' ')
            f.writelines(str(yolo.a_top[i]))
            f.writelines(' ')
            f.writelines(str(yolo.a_right[i]))
            f.writelines(' ')
            f.writelines(str(yolo.a_bottom[i]))
            f.writelines('\n')
    yolo.close_session()


if __name__ == '__main__':
    # detect_img(YOLO())
    detect_img_for_map(YOLO())



    # img_path = input('please input your image path:')
    # image = Image.open(img_path)
    # label = [[10,20,30,40,3]]
    # draw_label(image, label)
