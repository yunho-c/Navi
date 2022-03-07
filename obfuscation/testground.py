from matplotlib import pyplot as plt
import path
from cv2 import imread


from glob import glob

import os

data_dir = './dataset/CASIA1/1/'

files = glob(os.path.join(data_dir, "*_1_*.jpg"))
print(files)

a = imread('./dataset/CASIA1/2/002_1_2.jpg',0)

plt.imshow(a)
plt.show()

