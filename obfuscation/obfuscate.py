import numpy as np

from cv2 import imread, imwrite, cvtColor, COLOR_BGR2GRAY

from denormalize import denormalize
from fnc.segment import segment
from fnc.normalize import normalize
from detail_extraction import erase_non_iris, detail_extraction

# since this is the entry point, define parameters

# config file seems like great idea
# gotta split multiprocessing btwn matching & the rest though
# 아. 그냥 꺼놓을거지 ㅋ
eyelashes_thres = 80
use_multiprocess = False # off is faster ..
radial_res = 120 # 보통 싸이즈랑 비슷하게 가장. nyquist 해도 되고.
angular_res = 240
GAIN = 1#1.5

# TODO: Better gain equation. right now, i don't think it's norm- (or max-) conserving
      # because it only uses one of the norms not both

# TODO: 모양이 이상하다. There is an altering stripe pattern. 뭔가 잘못된 듯.
      # as noted somewhere else, this is caused by imperfect r-value gradient.
      # in areas with constant r, stripe pattern occurs.
      # right now, it's casued where there are imperfections in iris/pupil mask.

# radial_res = 960
# just for fun!

# angular_res = 480
# TODO: something funky happens when angular resolution is changed. probably bc it's hard-coded

# iris class 만들면 좋겠는데? vue/ahrs처럼 nested dot function 하고!
# 클래스 생성만해도 다 생긴다든지,
# iris = Iris(im=img)
# detail = iris.detail()
# template, mask = iris.normalize() # normalization의 dimension과 magnitude component 나누자

def obfuscate(img): # input argument is image. segmentation is done within.
    obfuscated = img.copy()
    # user | target
    c_iris, c_pupil, segmented = segment(img, eyelashes_thres, use_multiprocess)
    iris = erase_non_iris(c_iris, segmented)
    detail = detail_extraction(c_iris, segmented)
    template, mask, norm = normalize(detail.copy(), c_iris, c_pupil, radial_res, angular_res) # template과 mask가 딱 적절한데 it's a taken name. 뭐어때. # 인코딩된건 'e_' prefix로!
    denormalized = denormalize(c_iris, c_pupil, iris, template)
    g = (GAIN*norm/(np.max(denormalized) - np.min(denormalized)))

    # donor | source
    donor = imread('./dataset/CASIA1/3/003_1_1.jpg',0)#'./dataset/CASIA1/104/104_1_2.jpg')
    c_iri2, c_pupi2, segmente2 = segment(donor, eyelashes_thres, use_multiprocess)
    detai2 = detail_extraction(c_iri2, segmente2)
    templat2, mas2, nor2 = normalize(detai2.copy(), c_iri2, c_pupi2, radial_res, angular_res)
    donation = denormalize(c_iris, c_pupil, iris, templat2)
    g = (GAIN*norm/(np.max(donation) - np.min(donation)))
    
    obfuscated = obfuscated - np.nan_to_num(detail) + np.nan_to_num(g*donation)
    
    # TODO: Remember that datatype is important in NumPy. It caused a good amount of debugging (2-3hrs) so far.
    #       Note that maxval of 'obfuscated' exceeds 255 until cast into uint8. 
    #       Therefore, there's more work to be done around brightness and gain.
    
    return obfuscated.astype(np.uint8)


def obfuscate_color(img):
    # decolorize
    img_mono = cvtColor(img, COLOR_BGR2GRAY)
    img_color = img / np.stack((img_mono,)*3, axis=-1) # duplicates grayscale into 3 channels so it's equidimensional with BGR and thus multipliable

    # perform obfuscation
    img_obsf = obfuscate(img_mono)

    # recolorize
    img_rslt = img_color * np.stack((img_obsf,)*3, axis=-1)

    return img_rslt.astype(np.uint8)


def main():
    from matplotlib import pyplot as plt

    filename = './dataset/CASIA1/1/001_1_2.jpg'

    im = imread(filename, 0)
    plt.title('Original', fontweight ="bold")
    plt.imshow(im, cmap='plasma'); plt.show()

    im_obs = obfuscate(im)
    plt.title('Obfuscated', fontweight ="bold")
    plt.imshow(im_obs, cmap='plasma'); plt.show()
    
if __name__ == "__main__": main()

# input_fn 
# output_fn

