from calendar import c
import os

directory = "C:/Users/jhchee/Documents/make_dataset_whiteflies/annotations_combined_standardised_class"


for filename in os.listdir(directory):

    f = os.path.join(directory, filename)
    # opening the file in read mode
    file = open(f, "r")


    replacement = ""
    # using the for loop
    for line in file:
        #line = line.strip()
        changes = line.replace("<name>n02247216</name>", "<name>whitefly</name>")
        replacement = replacement + changes

    file.close()
    # opening the file in write mode
    fout = open(f, "w")
    fout.write(replacement)
    fout.close()

print("done")