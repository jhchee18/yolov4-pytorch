from calendar import c
import os

directory = ".\dataset\dataset_aphids\image/"

def change_file_name():
    for filename in os.listdir(directory):
    # Construct old file name
        source = directory + filename

        # Adding the count to the new file name and extension
        destination = directory + "aphids_" + filename

        # Renaming the file
        os.rename(source, destination)

    print("done")

change_file_name()
