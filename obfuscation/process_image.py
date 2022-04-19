from obfuscate import obfuscate, obfuscate_color
from detection import find_eyes
from cv2 import imread


def process_image(img, debug=False):

    # detect iris
    [eye1, eye2], coords = find_eyes(img, square=True)

    # perform obfuscation
    eye1_rslt = obfuscate_color(eye1)
    eye2_rslt = obfuscate_color(eye2)

    # paste back onto original image
    x, y, w, h = coords[0]
    img[x:x+w,y:y+h] = eye1_rslt
    x, y, w, h = coords[1] 
    img[x:x+w,y:y+h] = eye2_rslt

    # it'll be nice to be able to access intermediate representations, like segmentation results, etc
    if debug: pass

    return img


def main():
    filename = 'testing3.jpg'
    import cv2

    orig = imread(filename)
    cv2.imshow('original', orig)
    cv2.waitKey(0); cv2.destroyAllWindows()

    img = process_image(filename)
    cv2.imshow('obfuscated', img)
    cv2.waitKey(0); cv2.destroyAllWindows()

if __name__ == "__main__": main()