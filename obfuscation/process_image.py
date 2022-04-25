from obfuscate import obfuscate, obfuscate_color
from detection import find_eyes
from cv2 import imread


def process_image(img, debug=False):

    # detect eye regions
    eyes, coords = find_eyes(img, square=True)

    for index, eye in enumerate(eyes):
        # perform obfuscation
        result = obfuscate_color(eye)

        # paste back onto original image
        x, y, w, h = coords[index]
        img[x:x+w,y:y+h] = result

    # it'll be nice to be able to access intermediate representations, like segmentation results, etc
    if debug: pass

    return img


def main():
    filename = 'testing.jpg'
    import cv2

    orig = imread(filename)
    cv2.imshow('original', orig)
    cv2.waitKey(0); cv2.destroyAllWindows()

    img = process_image(imread(filename))
    cv2.imshow('obfuscated', img)
    cv2.waitKey(0); cv2.destroyAllWindows()

if __name__ == "__main__": main()