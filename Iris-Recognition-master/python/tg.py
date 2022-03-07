from matplotlib import pyplot as plt
import numpy as np

cir_iris = [141, 171, 100]
cir_pupil = [138, 173, 39]

cntr_diff = [-3, 2]
cntr_iris = [141, 171]
r_diff = -61
r_grid = np.zeros(shape=[280,320])

radial_res = 20

def draw_circle(img, cntr, r, val): 
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    dist = np.sqrt((x - cntr[1])**2 + (y - cntr[0])**2)
    return np.where(dist <= r, val, img)

def smart_r_grid(cir_iris, cir_pupil, iris_bool):
    cntr_iris = cir_iris[0:2]; r_iris = cir_iris[2]
    cntr_diff = np.array(cir_pupil[0:2]) - np.array(cir_iris[0:2]) 
    r_diff = cir_pupil[2] - cir_iris[2]

    r_grid = iris_bool.copy().astype(int)
    for i in range(20):
        c = 20-i
        r_grid = draw_circle(r_grid, 
                             cntr_iris+cntr_diff - cntr_diff*((c-1)/radial_res), 
                             r_iris - r_diff*((c-1)/radial_res), val=c)

    return r_grid

r_grid = smart_r_grid(cir_iris, cir_pupil, r_grid)
plt.imshow(r_grid)
plt.show()