from calendar import c
from fileinput import filename
import os
import string

# filename = "C:/Users/jhchee/Documents/dataset_leaf_scorch/valid/_annotations_2007.txt"

# filename = "C:/Users/jhchee/Documents/dataset_leaf_scorch/test/_annotations_imagesets.txt"

filename = "/home/howard/ur5_ws/src/drone_vision/dataset/dataset_leaf_and_dbm/annotations_test_leaf_and_dbm.txt"

# for generating test.txt/ train.txt/trainval.txt/val.txt in ImageSets/Main
def get_filename_from_annotations():

    f = filename
    # opening the file in read mode
    file = open(f, "r")

    replacement = ""
    # using the for loop
    for line in file:

        # line = line.strip()
        changes = line.split(".jpg", 1)[0]
        replacement = replacement + changes + "\n"

    file.close()
    # opening the file in write mode
    fout = open(f, "w")
    fout.write(replacement)
    fout.close()

    print("done")


def get_full_annotation_2007():

    f = filename
    # opening the file in read mode
    file = open(f, "r")

    replacement = ""
    # using the for loop
    for line in file:

        # line = line.strip()
        changes = "C:/Users/jhchee/Documents/yolov4-pytorch/VOCdevkit/VOC2007/JPEGImages/" + line
        replacement = replacement + changes

    file.close()
    # opening the file in write mode
    fout = open(f, "w")
    fout.write(replacement)
    fout.close()

    print("done")


def get_full_annotation_pc():

    f = filename
    # opening the file in read mode
    file = open(f, "r")

    replacement = ""
    # using the for loop
    for line in file:

        # line = line.strip()
        changes = "./dataset/dataset_leaf_and_dbm/test/" + line
        replacement = replacement + changes

    file.close()
    # opening the file in write mode
    fout = open(f, "w")
    fout.write(replacement)
    fout.close()

    print("done")


def attach_suffix():

    f = filename
    # opening the file in read mode
    file = open(f, "r")

    replacement = ""
    # using the for loop
    for line in file:

        # line = line.strip()
        changes = line.split("\n", 1)[0] + ".jpg\n"
        replacement = replacement + changes

    file.close()
    # opening the file in write mode
    fout = open(f, "w")
    fout.write(replacement)
    fout.close()

    print("done")


def replace_class_num():

    f = filename
    # opening the file in read mode
    file = open(f, "r")

    replacement = ""
    # using the for loop
    for line in file:

        # line = line.strip()
        changes = line.replace("dataset_leaf_scorch", "dataset_leaf_and_dbm")
        replacement = replacement + changes

    file.close()
    # opening the file in write mode
    fout = open(f, "w")
    fout.write(replacement)
    fout.close()

    print("done")


get_full_annotation_pc()
#replace_class_num()
#attach_suffix()