from matplotlib.pyplot import polar
import numpy as np

from fnc.normalize import normalize
from detail_extraction import erase_non_iris, detail_extraction

radial_res = 20
angular_res = 240


def draw_circle(img, cntr, r, val): 
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    dist = np.sqrt((x - cntr[1])**2 + (y - cntr[0])**2)
    return np.where(dist <= r, val, img)


def smart_r_grid(cir_iris, cir_pupil, iris_bool):
    cntr_iris = cir_iris[0:2]; r_iris = cir_iris[2]
    cntr_diff = np.array(cir_pupil[0:2]) - np.array(cir_iris[0:2]) 
    r_diff = cir_iris[2] - cir_pupil[2]

    r_grid = iris_bool.copy().astype(int)
    for i in range(20):
        c = 20-i
        r_grid = draw_circle(r_grid, 
                             cntr_iris+cntr_diff - cntr_diff*((c-1)/radial_res), 
                             r_iris - r_diff*((i+1)/radial_res), val=c)
        print(r_iris - r_diff*((i+1)/radial_res), c)

    return r_grid
    # pupil 문제로 인하여 (또는 boundary case에 대한 접근의 차이로 인하여) masked region에 노이즈가 생기는 듯 함.!

def denormalize(cir_iris, cir_pupil, im_iris, normalized_template):
    # iris = np.nan_to_num(im_iris).astype(bool) # boolean.
    iris = ~np.isnan(im_iris)
    # iris_nan = iris_bool.copy()

    # dense grid coordinate values
    y_grid, x_grid = np.mgrid[0:iris.shape[0], 0:iris.shape[1]]

    # if it's not iris, replace with 0
    # ideally NaN though
    y_grid = np.where(iris, y_grid, 0)
    x_grid = np.where(iris, x_grid, 0)

    # zero into the iris center -- relative coordinates
    cntr_iris = cir_iris[0:2]; r_iris = cir_iris[2]
    y_c_grid = y_grid - cntr_iris[0]
    # y_c_grid = -y_c_grid # flip signs
    x_c_grid = x_grid - cntr_iris[1]

    # use it to derive a matrix representing r and a matrix 
    # representing theta per each point in original space.
    r_grid = np.sqrt(y_c_grid**2 + x_c_grid**2)
    th_grid = np.arctan2(y_c_grid, x_c_grid)

    # scaling and discretization for coordinate alignment
    # r_grid = (radial_res/r_iris * r_grid).astype(int) # r -> y
    r_grid = smart_r_grid(cir_iris, cir_pupil, iris) # r -> y
    th_grid = (angular_res/(2*np.pi) * th_grid).astype(int) # th -> x

    # last-minute data adjustment: boundary case, zero-norm
    r_grid -= 1; th_grid -= 1
    th_grid += angular_res//2

    # remove background offset # TEMPORARY
    r_grid = np.where(iris, r_grid, 0)
    th_grid = np.where(iris, th_grid, 0)


    a = normalized_template[r_grid, th_grid]
    a = np.where(iris, a, 0)

    return a


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
    result = im1 - np.nan_to_num(detail)
    # test: add normalized -> denormalized detail from same file
    polar_array, noise_array = normalize(detail, cir_iris[1], cir_iris[0], cir_iris[2],
                                        cir_pupil[1], cir_pupil[0], cir_pupil[2],
                                        radial_res, angular_res)
    denormalized_detail = denormalize(cir_iris, cir_pupil, iris, polar_array)
    result = result + (5/np.max(denormalized_detail + np.min(denormalized_detail)))*denormalized_detail

    min, max = np.min(denormalized_detail), np.max(denormalized_detail)
    plt.title('Denormalized Detail: Radial Resolution={}, Angular Resolution: {}, Zmin: {}, Zmax: {}'.format(radial_res, angular_res, min, max), fontweight ="bold")
    plt.imshow(denormalized_detail)
    plt.show()

    plt.title('Result: Radial Resolution={}, Angular Resolution: {}'.format(radial_res, angular_res), fontweight ="bold")
    plt.imshow(result)
    plt.show()



    # test: add detail from another file
    # hamming distance
    # image comparison (50/50)

    # 음 몰라~!




if __name__ == "__main__": main()