#imports

import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt

#OpenCV Haar detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_eye.xml')
# cv2.samples.findFile()

def find_eyes(img, square=False):
    eyes = eyes_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=6)

    eyeImages = []; coords = []

    for eye in eyes:
        y,x,h,w = eye
        # if square: w,h = [min(w,h)-1]*2 #DEBUG # VERY naive implementation
        eyeImages.append(img[x:x+w,y:y+h])
        coords.append([x,y,w,h])

    return eyeImages, coords

#     tuple1 = (eye[0][0], eye[0][1])
#     tuple2 = (eye[0][1])
#     tuple3 = (eye[0][0], eye[0][0] + eye[0][2])
#     tuple4 = (eye[1][0], eye[1][1] + eye[1][3])
#     cv2.rectangle(img,tuple1,tuple2,(255, 0, 0),2)
#     # cv2.rectangle(img,tuple3,tuple4,(0, 255, 0),2)

#     cv2.imwrite("testingpics.jpg", img)

def main():    
    filename = cv2.imread("testing.jpg")
    [eye1, eye2], coords = find_eyes(filename, square=True)
    print(coords)
    plt.imshow(cv2.cvtColor(eye1, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.imshow(cv2.cvtColor(eye2, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == "__main__": main()