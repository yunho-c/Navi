import numpy as np

radial_res = 20
angular_res = 240

from detail_extraction import erase_non_iris

# TODO: account for non-centered pupil
# 애초에 normalize.py 저 개같은거 원리를 이해하지도 못하겠음
# 아마도 pupil은 잘라내고 남은 부분에 radial resolution을 맞추는걸듯
# 씁,, ㅠ 생각해보니 열심히 해도 1371 들으면 다들 나만큼 하겠다. 더잘하거나

# TODO: add scaling

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
    th_grid = angular_res/(2*np.pi) * th_grid.astype(int) # th -> x
    # TODO: !!!! pupil 안빼면 반띵돼버리네

    normalized_template[th_grid, r_grid]

    return 


def main():
    from matplotlib import pyplot as plt
    from fnc.segment import segment
    from cv2 import imread #, imwrite

    # read, segment, clean sample image
    im = imread('001_1_2.bmp', 0)
    eyelashes_thres = 80; use_multiprocess = False
    cir_iris, cir_pupil, imwithnoise = segment(im, eyelashes_thres, use_multiprocess)
    iris = erase_non_iris(cir_iris, imwithnoise)
    # test: add detail from another file


if __name__ == "__main__": main()