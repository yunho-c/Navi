from matplotlib import pyplot as plt
import path
from cv2 import imread

a = imread('../CASIA1/2/002_1_2.jpg',0)

plt.imshow(a)
plt.show()