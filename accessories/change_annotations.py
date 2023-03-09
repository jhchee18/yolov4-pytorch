from calendar import c
from fileinput import filename
import os
import string
import os
import numpy as np
from PIL import Image

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



label_directory = ".\dataset\dataset_aphids\label"
combined_annotation_filename = "./dataset/dataset_aphids/combined_annotation.txt"
image_directory = "./dataset/dataset_leaf_dbm_aphids/train_valid/"

def combine_annotation():
    combined_annotation_text = ""
    for filename in os.listdir(label_directory):

        f = os.path.join(label_directory, filename)
        file = open(f, "r")
        annotation = ""
        for line in file:
            changes = ""
            changes = " " + line.split("\n", 1)[0]
            annotation = annotation + changes
        combined_annotation_text = combined_annotation_text + image_directory + filename.split(".txt", 1)[0] + ".jpg" + annotation + "\n"
        file.close()

    fout = open(combined_annotation_filename, "w")
    fout.write(combined_annotation_text)
    fout.close()
    print("done")



def generate_test_image_names():
    test_image_directory = "./dataset/dataset_leaf_and_dbm/test_15122022/"
    combined_annotation_filename = "./dataset/dataset_leaf_and_dbm/annotations_test_leaf_and_dbm_15122022.txt"
    combined_names_text = ""
    for filename in os.listdir(test_image_directory):
        combined_names_text = combined_names_text + test_image_directory + filename + "\n"
    fout = open(combined_annotation_filename, "w")
    fout.write(combined_names_text)
    fout.close()
    print("done generate_test_image_names")


train_annotation_file = ".\dataset\dataset_aphids/train_annotation.txt"
valid_annotation_file = ".\dataset\dataset_aphids/valid_annotation.txt"

def split_annotation_train_and_valid():
    f = combined_annotation_filename
    # opening the file in read mode
    file = open(f, "r")

    train_annotation = ""
    valid_annotation = ""
    counter = 0
    # using the for loop
    for line in file:
        counter += 1
        changes = line
        if (counter % 5 != 0):
            train_annotation = train_annotation + changes
        else:
            valid_annotation = valid_annotation + changes

    file.close()
    # opening the file in write mode
    ftrainout = open(train_annotation_file, "w")
    ftrainout.write(train_annotation)
    ftrainout.close()

    fvalidout = open(valid_annotation_file, "w")
    fvalidout.write(valid_annotation)
    fvalidout.close()

    print("done")



def move_file():
    origin = "./dataset/dataset_leaf_and_dbm/train_valid/"
    dest = "./dataset/dataset_leaf_and_dbm/removed/"
    combined_annotation_file = "./dataset/dataset_leaf_and_dbm/annotations_train_leaf_and_dbm.txt"
    removed_annotation_file = "./dataset/dataset_leaf_and_dbm/removed_annotations_train.txt"

    substring1 = ",2 "
    substring2 = ",3 "
    substring3 = ",2\n"
    substring4 = ",3\n"
    f = combined_annotation_file
    # opening the file in read mode
    file = open(f, "r")

    removed_annotation_text = ""
    remain_annotation_text = ""
    for line in file:
        if (line.find(substring1) != -1 or line.find(substring2) != -1 or line.find(substring3) != -1 or line.find(substring4) != -1):
            removed_annotation_text = removed_annotation_text + line
            filename = line.split(" ", 1)[0].split("/", 5)[-1]
            os.rename(origin + filename, dest + filename)
        else:
            remain_annotation_text = remain_annotation_text + line


    file.close()
    # opening the file in write mode
    fremoved = open(removed_annotation_file, "w")
    fremoved.write(removed_annotation_text)
    fremoved.close()

    fremained = open(combined_annotation_file, "w")
    fremained.write(remain_annotation_text)
    fremained.close()

    print("done moving files!")

def move_file_without_annotation():
    origin = "./dataset/dataset_aphids/image/"
    dest = "./dataset/dataset_aphids/removed/"
    combined_annotation_file = "./dataset/dataset_aphids/combined_annotation.txt"

    f = combined_annotation_file
    # opening the file in read mode
    file = open(f, "r")

    for line in file:
        filename = line.split(" ", 1)[0].split("/", 5)[-1]
        os.rename(origin + filename, dest + filename)

    file.close()
    print("done moving files!")


#get_full_annotation_pc()
#replace_class_num()
#attach_suffix()
#ombine_annotation()
#split_annotation_train_and_valid()
#generate_test_image_names()
#move_file()
move_file_without_annotation()