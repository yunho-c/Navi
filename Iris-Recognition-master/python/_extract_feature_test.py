# import argparse
import matplotlib.pyplot as plt


from fnc.extractFeature import extractFeature
from fnc.matching import matching

# parser = argparse.ArgumentParser()

# parser.add_argument("--file", type=str,
#                     help="Path to the file that you want to verify.")

# args = parser.parse_args()

# filename = args.file

filename = '001_1_2.bmp'

print('>>> Start verifying {}\n'.format(filename))
template, mask, file = extractFeature(filename, use_multiprocess=False)

# a = mask
a = template

c = plt.imshow(a)   #cmap ='Greens', vmin = z_min, vmax = z_max,
                    #extent =[x.min(), x.max(), y.min(), y.max()],
                    #interpolation ='nearest', origin ='lower')

# plt.title('Extracted Feature', fontweight ="bold")

plt.show()

print('good')