from calendar import c
from fileinput import filename
import os
import string

filename = "C:/Users/jhchee/Documents/dataset_dbm/train/_annotations_2007.txt"


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
        changes = line.replace("Diamondback", "C:/Users/jhchee/Documents/yolov4-pytorch/VOCdevkit/VOC2007/JPEGImages/Diamondback")
        replacement = replacement + changes

    file.close()
    # opening the file in write mode
    fout = open(f, "w")
    fout.write(replacement)
    fout.close()

    print("done")

get_full_annotation_2007()