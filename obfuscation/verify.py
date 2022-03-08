##-----------------------------------------------------------------------------
##  Import
##-----------------------------------------------------------------------------
import argparse
from time import time

from fnc.extractFeature import extractFeature
from fnc.matching import matching

#------------------------------------------------------------------------------
#	Argument parsing
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--file", type=str, default='',
                    help="Path to the file that you want to verify.")

parser.add_argument("--temp_dir", type=str, default="./templates/CASIA1/",
					help="Path to the directory containing templates.")

parser.add_argument("--thres", type=float, default=0.38,
					help="Threshold for matching.")

args = parser.parse_args()
args.file = './dataset/CASIA1/8/008_1_2.jpg'# DEBUG


##-----------------------------------------------------------------------------
##  Execution
##-----------------------------------------------------------------------------
# Extract feature
def main():
	start = time()
	print('>>> Start verifying {}\n'.format(args.file))
	template, mask, file = extractFeature(args.file, use_multiprocess=True)


	# Matching
	print('matching is reached')
	result = matching(template, mask, args.temp_dir, args.thres)

	if result == -1:
		print('>>> No registered sample.')

	elif result == 0:
		print('>>> No sample matched.')

	else:
		print('>>> {} samples matched (descending reliability):'.format(len(result)))
		for res in result:
			print("\t", res)


	# Time measure
	end = time()
	print('\n>>> Verification time: {} [s]\n'.format(end - start))

if __name__ == "__main__": main()