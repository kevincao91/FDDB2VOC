import os
import cv2
import numpy as np
from xml.dom.minidom import Document

root_dir = "/home/kevin/文档/"
fddb_root_dir = root_dir + "FDDB/"
voc_root_dir = "./VOCFace/"
origimage_dir = fddb_root_dir + "originalPics/"
fddb_annotation_dir = fddb_root_dir + "FDDB-folds/"
voc_images_dir = voc_root_dir + "JPEGImages/"
voc_txt_dir = voc_root_dir + "ImageSets/"
voc_temp_txt_dir = voc_root_dir + "ImageSets/temp/"
voc_annotations_dir = voc_root_dir + "Annotations/"
convert2rects = True
bsavexmlanno = True
bsavetxtanno = True
windowsshow = False
copyimg = False


def show_annotations():
    if not os.path.exists(voc_root_dir):
        os.mkdir(voc_root_dir)
    if not os.path.exists(voc_annotations_dir):
        os.mkdir(voc_annotations_dir)
    if not os.path.exists(voc_txt_dir):
        os.mkdir(voc_txt_dir)
    if not os.path.exists(voc_temp_txt_dir):
        os.mkdir(voc_temp_txt_dir)
    if not os.path.exists(voc_images_dir):
        os.mkdir(voc_images_dir)
    for i in range(10):
        fddb_annotation_filepath = fddb_annotation_dir + "/FDDB-fold-%0*d-ellipseList.txt" % (2, i + 1)
        fddb_annotation_file = open(fddb_annotation_filepath)
        while True:
            filename = fddb_annotation_file.readline()[:-1] + ".jpg"
            if not filename:
                break
            line = fddb_annotation_file.readline()
            if not line:
                break
            print(filename)
            facenum = int(line)
            img = cv2.imread(os.path.join(origimage_dir, filename))
            filename = filename.replace('/', '_')
            if copyimg:
                cv2.imwrite(os.path.join(voc_images_dir, filename), img)
            w = img.shape[1]
            h = img.shape[0]
            if bsavetxtanno:
                labelpath = voc_temp_txt_dir + "/" + filename.replace('/', '_')[:-3] + "txt"
                labelfile = open(labelpath, 'w')
            if bsavexmlanno:
                xmlpath = voc_annotations_dir + "/" + filename.replace('/', '_')[:-3] + "txt"
                xmlpath = xmlpath[:-3] + "xml"
                doc = Document()
                annotation = doc.createElement('annotation')
                doc.appendChild(annotation)
                folder = doc.createElement('folder')
                folder_name = doc.createTextNode('fddb')
                folder.appendChild(folder_name)
                annotation.appendChild(folder)
                filenamenode = doc.createElement('filename')
                filename_name = doc.createTextNode(filename)
                filenamenode.appendChild(filename_name)
                annotation.appendChild(filenamenode)
                source = doc.createElement('source')
                annotation.appendChild(source)
                database = doc.createElement('database')
                database.appendChild(doc.createTextNode('FDDB Database'))
                source.appendChild(database)
                annotation_s = doc.createElement('annotation')
                annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
                source.appendChild(annotation_s)
                image = doc.createElement('image')
                image.appendChild(doc.createTextNode('flickr'))
                source.appendChild(image)
                flickrid = doc.createElement('flickrid')
                flickrid.appendChild(doc.createTextNode('-1'))
                source.appendChild(flickrid)
                owner = doc.createElement('owner')
                annotation.appendChild(owner)
                flickrid_o = doc.createElement('flickrid')
                flickrid_o.appendChild(doc.createTextNode('Kevin'))
                owner.appendChild(flickrid_o)
                name_o = doc.createElement('name')
                name_o.appendChild(doc.createTextNode('Kevin'))
                owner.appendChild(name_o)
                size = doc.createElement('size')
                annotation.appendChild(size)
                width = doc.createElement('width')
                width.appendChild(doc.createTextNode(str(img.shape[1])))
                height = doc.createElement('height')
                height.appendChild(doc.createTextNode(str(img.shape[0])))
                depth = doc.createElement('depth')
                depth.appendChild(doc.createTextNode(str(img.shape[2])))
                size.appendChild(width)
                size.appendChild(height)
                size.appendChild(depth)
                segmented = doc.createElement('segmented')
                segmented.appendChild(doc.createTextNode('0'))
                annotation.appendChild(segmented)
            for j in range(facenum):
                line = fddb_annotation_file.readline().strip().split()
                major_axis_radius = float(line[0])
                minor_axis_radius = float(line[1])
                angle = float(line[2])
                center_x = float(line[3])
                center_y = float(line[4])
                score = float(line[5])
                angle = angle / 3.1415926 * 180
                cv2.ellipse(img, (int(center_x), int(center_y)),
                            (int(major_axis_radius), int(minor_axis_radius)), angle, 0., 360., (255, 0, 0))
                if convert2rects:
                    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                    cv2.ellipse(mask, (int(center_x), int(center_y)),
                                (int(major_axis_radius), int(minor_axis_radius)), angle, 0., 360., (255, 255, 255))
                    # cv2.imshow("mask",mask)
                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

                    # 选第一个框
                    r = cv2.boundingRect(contours[0])
                    x_min = r[0]
                    y_min = r[1]
                    x_max = r[0] + r[2]
                    y_max = r[1] + r[3]
                    xcenter = r[0] + r[2] / 2
                    ycenter = r[1] + r[3] / 2
                    if bsavetxtanno:
                        labelline = "0" + "\t" + str(xcenter * 1.0 / w) + '\t' + str(
                            ycenter * 1.0 / h) + '\t' + str(r[2] * 1.0 / w) + '\t' + str(r[3] * 1.0 / h) + '\n'
                        labelfile.write(labelline)
                    if bsavexmlanno:
                        object = doc.createElement('object')
                        annotation.appendChild(object)
                        object_name = doc.createElement('name')
                        object_name.appendChild(doc.createTextNode('face'))
                        object.appendChild(object_name)
                        pose = doc.createElement('pose')
                        pose.appendChild(doc.createTextNode('Unspecified'))
                        object.appendChild(pose)
                        truncated = doc.createElement('truncated')
                        truncated.appendChild(doc.createTextNode('1'))
                        object.appendChild(truncated)
                        difficult = doc.createElement('difficult')
                        difficult.appendChild(doc.createTextNode('0'))
                        object.appendChild(difficult)
                        bndbox = doc.createElement('bndbox')
                        object.appendChild(bndbox)
                        xmin = doc.createElement('xmin')
                        xmin.appendChild(doc.createTextNode(str(x_min)))
                        bndbox.appendChild(xmin)
                        ymin = doc.createElement('ymin')
                        ymin.appendChild(doc.createTextNode(str(y_min)))
                        bndbox.appendChild(ymin)
                        xmax = doc.createElement('xmax')
                        xmax.appendChild(doc.createTextNode(str(x_max)))
                        bndbox.appendChild(xmax)
                        ymax = doc.createElement('ymax')
                        ymax.appendChild(doc.createTextNode(str(y_max)))
                        bndbox.appendChild(ymax)
                    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255))
            if bsavetxtanno:
                labelfile.close()
            if bsavexmlanno:
                f = open(xmlpath, "w")
                f.write(doc.toprettyxml(indent=''))
                f.close()
            if windowsshow:
                cv2.imshow("img", img)
                cv2.waitKey()


def generatevocsets(trainratio=0.6, valratio=0.2, testratio=0.2):
    voc_main_txt_dir = os.path.join(voc_txt_dir, 'Main')
    if not os.path.exists(voc_main_txt_dir):
        os.mkdir(voc_main_txt_dir)
    ftrain = open(os.path.join(voc_main_txt_dir, 'train.txt'), 'w')
    fval = open(os.path.join(voc_main_txt_dir, 'val.txt'), 'w')
    ftrainval = open(os.path.join(voc_main_txt_dir, 'trainval.txt'), 'w')
    ftest = open(os.path.join(voc_main_txt_dir, 'test.txt'), 'w')
    files = os.listdir(voc_temp_txt_dir)

    train_point = int(len(files) * trainratio)
    val_point = train_point + int(len(files) * valratio)
    test_point = val_point + int(len(files) * testratio)
    for i in range(len(files)):
        imgfilename = files[i][:-4]
        if i < train_point:
            ftrain.write(imgfilename + "\n")
            ftrainval.write(imgfilename + "\n")
        elif i < val_point:
            fval.write(imgfilename + "\n")
            ftrainval.write(imgfilename + "\n")
        elif i < test_point:
            ftest.write(imgfilename + "\n")
    ftrain.close()
    fval.close()
    ftrainval.close()
    ftest.close()


if __name__ == "__main__":
    show_annotations()
    generatevocsets()

