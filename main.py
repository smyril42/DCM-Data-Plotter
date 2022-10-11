import pydicom
import matplotlib.pyplot as plt
import os

# creating a path-list for every file in the directory /012_fmre_40Hz_SS_11sl_TR1200
path = '012_fmre_40Hz_SS_11sl_TR1200'
file_list = sorted([f.path for f in os.scandir(path)])


i = 0
ds = pydicom.dcmread(file_list[i])

plt.imshow(ds.pixel_array)
plt.show()
