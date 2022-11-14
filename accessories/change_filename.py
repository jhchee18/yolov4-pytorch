from calendar import c
import os

directory = "C:\Users\jhchee\Documents\dataset_dbm/test"

def change_file_name():
    for filename in os.listdir(directory):
    # Construct old file name
        source = directory + filename

        # Adding the count to the new file name and extension
        destination = source[0:-5] + ".jpg"

        # Renaming the file
        os.rename(source, destination)

    print("done")


