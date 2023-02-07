# git checkout nicco

import pandas as pd
import os
import re
from PIL import Image
import numpy as np

# get folder names --> class names
fldrs = os.listdir("data/Kather_texture_2016_image_tiles_5000")
cls_names = ([re.sub(r"\d+", "", cls) .lstrip("_") for cls in fldrs])

# read the images to array
data_array = raw_dat_array = []
for fldr in fldrs:
    print(fldr)
    fldr_path = "data/Kather_texture_2016_image_tiles_5000/" + fldr
    files = os.listdir(fldr_path)
    tmp = raw_array = []
    for file in files:
        img = Image.open(fldr_path + "/" + file)
        imarray = np.array(img)
        raw_array.append(img)
        tmp.append(imarray)
    data_array.append(tmp)
    raw_dat_array.append(raw_array)

