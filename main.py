# import the necessary packages
import numpy as np
import cv2
import os
from xml.dom.minidom import Document


def oval2box(image, center_x, center_y, major_axis_radius, minor_axis_radius, angle, w, h):
    mask = np.zeros((w, h, 3), dtype=np.uint8)
    cv2.ellipse(mask, (center_x, center_y), (major_axis_radius, minor_axis_radius), angle, 0, 360, (255, 255, 255), 1)
    # cv2.imshow("mask", mask)
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for k in range(len(contours) - 2):
        r = cv2.boundingRect(contours[k])
        x_min = r[0]
        y_min = r[1]
        x_max = r[0] + r[2]
        y_max = r[1] + r[3]
        xcenter = r[0] + r[2] / 2
        ycenter = r[1] + r[3] / 2
        labelline = "0" + "\t" + str(xcenter * 1.0 / w) + '\t' + str(ycenter * 1.0 / h) + '\t' + str(
            r[2] * 1.0 / w) + '\t' + str(r[3] * 1.0 / h) + '\n'
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 1)


def main():
    data_set_img_path = '/home/kevin/文档/FDDB/originalPics/'
    data_set_inf_path = '/home/kevin/文档/FDDB/FDDB-folds/'

    # for i in range(0):
    i = 1
    print(i)
    image_path = os.path.join(data_set_img_path, '2002/08/26/big/img_265' + '.jpg')
    information_path = []
    information_num = 3
    information = [[67.363819, 44.511485, -1.476417, 105.249970, 87.209036],
                   [41.936870, 27.064477, 1.471906, 184.070915, 129.345601],
                   [70.993052, 43.355200, 1.370217, 340.894300, 117.498951]]

    # Read image
    image = cv2.imread(image_path)

    w = image.shape[1]
    h = image.shape[0]
    print(w, h)

    # Copy image as original
    org = image.copy()

    # Draw parameters
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw
    for i in range(information_num):
        major_axis_radius, minor_axis_radius, angle, center_x, center_y = information[i]
        center_x = round(center_x)
        center_y = round(center_y)
        major_axis_radius = round(major_axis_radius)
        minor_axis_radius = round(minor_axis_radius)
        angle = angle / (2 * np.pi) * 360
        cv2.ellipse(org, (center_x, center_y), (major_axis_radius, minor_axis_radius), angle, 0, 360, (255, 255, 255),
                    1)  # 画椭圆

        # Draw
        oval2box(image, center_x, center_y, major_axis_radius, minor_axis_radius, angle, w, h)

    # Show image
    cv2.imshow('Original', org)
    cv2.imshow('Done', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def test():
    data_set_img_path = '/home/kevin/文档/FDDB/originalPics/'
    data_set_inf_path = '/home/kevin/文档/FDDB/FDDB-folds/'
    annotations_dir = './Annotations/'

    if not os.path.exists(annotations_dir):
        os.mkdir(annotations_dir)

    # for i in range(0):
    i = 1
    print(i)
    image_path = os.path.join(data_set_img_path, '2002/08/26/big/img_265' + '.jpg')
    information_path = []
    information_num = 3
    information = [[67.363819, 44.511485, -1.476417, 105.249970, 87.209036],
                   [41.936870, 27.064477, 1.471906, 184.070915, 129.345601],
                   [70.993052, 43.355200, 1.370217, 340.894300, 117.498951]]

    # Read image
    image = cv2.imread(image_path)

    w = image.shape[1]
    h = image.shape[0]
    print(w, h)

    # Copy image as original
    org = image.copy()

    # Draw
    for i in range(information_num):
        major_axis_radius, minor_axis_radius, angle, center_x, center_y = information[i]
        center_x = round(center_x)
        center_y = round(center_y)
        major_axis_radius = round(major_axis_radius)
        minor_axis_radius = round(minor_axis_radius)
        angle = angle / (2 * np.pi) * 360
        cv2.ellipse(org, (center_x, center_y), (major_axis_radius, minor_axis_radius), angle, 0, 360, (255, 255, 255),
                    1)  # 画椭圆
        cv2.ellipse(image, (center_x, center_y), (major_axis_radius, minor_axis_radius), angle, 0, 360, (255, 255, 255),
                    1)  # 画椭圆

        mask = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.ellipse(mask, (center_x, center_y), (major_axis_radius, minor_axis_radius), angle, 0, 360, (255, 255, 255),
                    1)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        r = cv2.boundingRect(contours[0])
        x_min = r[0]
        y_min = r[1]
        x_max = r[0] + r[2]
        y_max = r[1] + r[3]
        xcenter = r[0] + r[2] / 2
        ycenter = r[1] + r[3] / 2

        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 1)

    # Show image
    cv2.imshow('Original', org)
    # cv2.imshow("Mask", mask)
    cv2.imshow('Done', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test()
    # main()
