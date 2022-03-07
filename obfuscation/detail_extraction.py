# NO PREMATURE OPTIMIZATION

import numpy as np
import skimage.filters
from matplotlib import pyplot as plt
from nan_conserving_gaussian import nan_conserving_gaussian

# BLUR PARAMETERS
SIGMA = 10 # 특정 숫자 필요한가?
TRUNCATE = 3.5 # 특정 숫자 필요한가?

def circle_mask(cntr, r, imsize):
    y, x = np.ogrid[:imsize[0], :imsize[1]]
    dist_from_cntr = np.sqrt((x - cntr[1])**2 + (y - cntr[0])**2)

    mask = dist_from_cntr <= r
    return mask

def erase_non_iris(cir_iris, img):
    # no eyelash in imwithnoise, please
    # if distance to iris center is greater than radius: remove
    mask = circle_mask(cir_iris[:2], cir_iris[2], img.shape)
    return np.where(mask, img, np.nan)


def blur_gaussian(img):
    # img = np.nan_to_num(img)
    # return skimage.filters.gaussian(img, sigma=(SIGMA, SIGMA), truncate=1.0*TRUNCATE, multichannel=True)
    return nan_conserving_gaussian(img, sigma=SIGMA)

def blur_radial(img):
    return

def remove_average_flat(img):
    new_img = img - np.nanmean(img)
    return new_img

def detail_extraction(cir_iris, img):  # not implemented
    # input: cir_iris, image
    # output: detail, original with detail removed -> result
    iris = erase_non_iris(cir_iris, img)
    blrd = blur_gaussian(iris)
    detail = iris - blrd
    # detail = remove_average_flat(detail)
    return detail

def main():
    from cv2 import imread, imwrite
    from fnc.segment import segment
    im = imread('001_1_2.bmp', 0) # filename
    eyelashes_thres = 80; use_multiprocess = False

    cir_iris, cir_pupil, imwithnoise = segment(im, eyelashes_thres, use_multiprocess)
    iris = erase_non_iris(cir_iris, imwithnoise)
    blrd = blur_gaussian(iris)
    detail = iris - blrd
    detail = remove_average_flat(detail)
    result = im - np.nan_to_num(detail)
    
    plt.title('Blurred, Gaussian, Sigma: {}, Truncate: {}'.format(SIGMA, TRUNCATE), fontweight ="bold")
    plt.imshow(blrd)#, vmin=0, vmax=255)
    plt.show()
    plt.title('Detail, Gaussian, Sigma: {}, Truncate: {}'.format(SIGMA, TRUNCATE), fontweight ="bold")
    plt.imshow(detail)#, vmin=0, vmax=255)
    plt.show()
    plt.title('Result, Gaussian, Sigma: {}, Truncate: {}'.format(SIGMA, TRUNCATE), fontweight ="bold")
    plt.imshow(result)
    plt.show()

    imwrite('001_1_2_detail_removed.jpg', result)

if __name__ == "__main__": main()