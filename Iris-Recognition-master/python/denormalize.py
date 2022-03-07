import numpy as np

from fnc.normalize import normalize
from detail_extraction import erase_non_iris, detail_extraction

radial_res = 20
angular_res = 240


def draw_circle(img, cntr, r, val): 
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    dist = np.sqrt((x - cntr[1])**2 + (y - cntr[0])**2)
    return np.where(dist <= r, img, val)

def smart_r_grid(cir_iris, cir_pupil, iris_bool):
    cntr_iris =  cir_iris[0:2]; r_iris = cir_iris[2]
    cntr_diff = cir_pupil[0:2] - cir_iris[0:2]; r_diff = cir_pupil[2] - cir_iris[2]

    r_grid = iris_bool.copy().astype(int)
    for i in range(20):
        r_grid = draw_circle(r_grid, 
                                cntr_iris + cntr_diff*((i+1)/radial_res), 
                                r_iris + r_diff*((i+1)/radial_res), val=i)
    return r_grid


def denormalize(cir_iris, cir_pupil, im_iris, normalized_template):
    # iris = np.array(im_iris, dtype=bool) # note: boolean
    iris = im_iris.astype(bool) # 이거 될까?

    # dense grid coordinate values
    y_grid, x_grid = np.mgrid[0:iris.shape[0], 0:iris.shape[1]]

    # if it's not iris, replace with NaN
    y_grid = np.where(not iris, y_grid, np.nan) 
    x_grid = np.where(not iris, x_grid, np.nan)

    # zero into the iris center -- relative coordinates
    cntr_iris = cir_iris[0:2]; r_iris = cir_iris[2]
    y_c_grid = y_grid - cntr_iris[1]
    x_c_grid = x_grid - cntr_iris[0]

    # use it to derive a matrix representing r and a matrix 
    # representing theta per each point in original space.
    r_grid = np.sqrt(y_c_grid**2 + x_c_grid**2)
    th_grid = np.arctan2(y_c_grid/x_c_grid)

    # scaling and discretization for coordinate alignment
    r_grid = radial_res/r_iris * r_grid.astype(int) # r -> y
    effective_r_grid = smart_r_grid(cir_iris, cir_pupil, iris) # r -> y
    th_grid = angular_res/(2*np.pi) * th_grid.astype(int) # th -> x

    normalized_template[th_grid, r_grid]

    return 


def main():
    from matplotlib import pyplot as plt
    from fnc.segment import segment
    from cv2 import imread #, imwrite

    # read, segment, clean sample image
    im1 = imread('001_1_2.bmp', 0)
    eyelashes_thres = 80; use_multiprocess = False
    cir_iris, cir_pupil, imwithnoise = segment(im1, eyelashes_thres, use_multiprocess)
    iris = erase_non_iris(cir_iris, imwithnoise)
    detail = detail_extraction(cir_iris, imwithnoise)
    # test: add normalized -> denormalized detail from same file
    result = ''
    normalize
    # test: add detail from another file
    # hamming distance
    # image comparison (50/50)

    # 음 몰라~!




if __name__ == "__main__": main()