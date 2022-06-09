#!/usr/env python3

import os
import shutil

for file in os.listdir("make_video"):
    file_name = file.zfill(len("000.png"))
    shutil.move(f"make_video/{file}", f"make_video/{file_name}")
