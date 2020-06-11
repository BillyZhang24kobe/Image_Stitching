# image stitching main implementation

import cv2
from matplotlib import pyplot as plt
from utils import *
import argparse
from stitching import *

parser = argparse.ArgumentParser(description="Awesome Image Stitcher")
parser.add_argument("-i", "--images", help="a sequence of images to be stitched", nargs="+",
                    action=required_length(3, 5))
parser.add_argument("-b", "--blending", help="blending techniques to be selected for the final panaroma",
                    choices=['avg', 'alpha', 'feathering', 'pyramid'])
parser.add_argument("-sp", "--scalePercent", help="scale percentage for resizing the images", type=int,
                    choices=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], default=100)
parser.add_argument("-gc", "--gainCompensation", help="adopts gain compensation for the final output image",
                    action="store_true")
parser.add_argument("--drawInliers", help="draw the inliers while creating the panaroma and store the results",
                    action="store_true")

if __name__ == '__main__':
    args = parser.parse_args()

    if not args.images:
        print("No images are selected for stitching. Program terminated!")
        print("Please run the program with '-h' flag to check the usage information!")
        exit(1)

    if not args.blending:
        print("No blending techniques are selected. Program terminated!")
        print("Please run the program with '-h' flag to check the usage information!")
        exit(1)

    imgs = args.images

    # enforce the input order -> increasing sequence order
    sorted_imgs = sorted(imgs)

    img_list_origin = []
    for i in sorted_imgs:
        img = cv2.cvtColor(cv2.imread(i, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        img_list_origin.append(img)

    print("Resizing images according to the scale percent")
    img_list = resize_img(img_list_origin, args.scalePercent)  # resize images to accelerate the computational speed


    print("Start stitching from left to right...")
    result = image_stitching(img_list, args.blending, args.gainCompensation, args.drawInliers)

    print('Done stitching!')
    plt.figure()
    plt.imshow(result)
    plt.xticks([]), plt.yticks([])
    plt.show()
