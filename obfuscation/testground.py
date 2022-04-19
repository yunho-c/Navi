import cv2
from matplotlib import pyplot as plt

filename = 'testing.jpg'
a = cv2.imread(filename)

cv2.imshow('asdf', a)
cv2.waitKey(0)
  

cv2.destroyAllWindows()