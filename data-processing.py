import os
import shutil


direct = "C:\\Users\\Mordor\Documents\\GitHub\\DL-Project\\tiny-imagenet-200\\val\\"

class_file = "C:\\Users\\Mordor\Documents\\GitHub\\DL-Project\\tiny-imagenet-200\\wnids.txt"


class_file = open(class_file, "r")

classes = class_file.readlines()


# Create directories
for class_name in classes:
    class_name = class_name[0:-1]
    if not os.path.exists(direct + class_name):
        os.makedirs(direct + class_name)




file = "val_annotations.txt"

instruction_file = open(direct+file, "r")

instruction_lines = instruction_file.readlines()


instruction_file.close()

for line in instruction_lines:
    keywords = line.split()
    image = keywords[0]
    clas = keywords[1]
    shutil.move(direct + "images\\" + image, direct + clas + "\\" + image)











