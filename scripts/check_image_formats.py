import os

import cv2


def check_images(s_dir, ext_list):
    bad_images = []
    bad_ext = []
    s_list = os.listdir(s_dir)
    for klass in s_list:
        klass_path = os.path.join(s_dir, klass)
        print("processing class directory ", klass)
        if os.path.isdir(klass_path):
            file_list = os.listdir(klass_path)
            for f in file_list:
                f_path = os.path.join(klass_path, f)
                index = f.rfind(".")
                ext = f[index + 1 :].lower()
                if ext not in ext_list:
                    print("file ", f_path, " has an invalid extension ", ext)
                    bad_ext.append(f_path)
                if os.path.isfile(f_path):
                    try:
                        img = cv2.imread(f_path)
                        shape = img.shape
                    except:
                        print("file ", f_path, " is not a valid image file")
                        bad_images.append(f_path)
                else:
                    print(
                        "*** fatal error, you a sub directory ",
                        f,
                        " in class directory ",
                        klass,
                    )
        else:
            print(
                "*** WARNING*** you have files in ",
                s_dir,
                " it should only contain sub directories",
            )
    return bad_images, bad_ext


source_dir = r"tensorflow-great-barrier-reef/train_images/"
good_exts = ["jpg", "png", "jpeg", "gif", "bmp"]  # list of acceptable extensions
bad_file_list, bad_ext_list = check_images(source_dir, good_exts)
if len(bad_file_list) != 0:
    print("improper image files are listed below")
    for i in range(len(bad_file_list)):
        print(bad_file_list[i])
else:
    print(" no improper image files were found")
