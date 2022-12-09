from calendar import c
import os

directory = "./dataset/dataset_drone_leaf_08_12_2022/label/"

def change_class_name():
    for filename in os.listdir(directory):

        f = os.path.join(directory, filename)
        # opening the file in read mode
        file = open(f, "r")


        replacement = ""
        # using the for loop
        for line in file:
            #line = line.strip()
            #changes = line.replace("C:/Users/jhchee/Documents/yolo-v4-pytorch/VOCdevkit/VOC2007/JPEGImages/", "./dataset/label_from_img/stem/")
            changes = "1" + line[1:]
            replacement = replacement + changes

        file.close()
        # opening the file in write mode
        fout = open(f, "w")
        fout.write(replacement)
        fout.close()

    print("done")

change_class_name()