from calendar import c
import os

directory = ".\dataset\dataset_drone_leaf_08_12_2022\label/"

def change_file_name():
    for filename in os.listdir(directory):
    # Construct old file name
        source = directory + filename

        # Adding the count to the new file name and extension
        destination = directory + "fungi_08122022_" + filename

        # Renaming the file
        os.rename(source, destination)

    print("done")

change_file_name()
