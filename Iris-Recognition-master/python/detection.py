import cv2 as cv
import numpy as np
import copy
import matplotlib.pyplot as plt
#importss
def read_image(image):
    img = cv.imread(image)
    outImg = img.copy()

    #OpenCV Haar detector
    face_detection = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_detection = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")
    eye = eye_detection.detectMultiScale(img, scaleFactor=1.3, minNeighbors=6)
    
    #make it possible for multiple eyes
    eyeImage1 = img[eye[0][1]:eye[0][1]+eye[0][3], eye[0][0]:eye[0][0]+eye[0][2]]
    eyeImage2 = img[eye[1][1]:eye[1][1]+eye[1][3], eye[1][0]:eye[1][0]+eye[1][2]]
        

    return eyeImage1, eyeImage2, eye

#     tuple1 = (eye[0][0], eye[0][1])
#     tuple2 = (eye[0][1])
#     tuple3 = (eye[0][0], eye[0][0] + eye[0][2])
#     tuple4 = (eye[1][0], eye[1][1] + eye[1][3])
#     cv.rectangle(img,tuple1,tuple2,(255, 0, 0),2)
#     # cv.rectangle(img,tuple3,tuple4,(0, 255, 0),2)

#     cv.imwrite("testingpics.jpg", img)

    

filename = "testing.jpg"
print(read_image(filename))

# cv.waitKey(0)
# cv.destroyAllWindows()
# plt.show()
