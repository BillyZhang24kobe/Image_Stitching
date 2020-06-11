import math
import numpy as np
import cv2
import argparse

def required_length(nmin,nmax):
    """
    custom action function in argparse for required length of nargs
    @param nmin: minimum argument length
    @param nmax: maximum argument length
    @return: required length object
    """
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if not nmin<=len(values)<=nmax:
                msg='argument "{images}" requires between {nmin} and {nmax} images to be stitched'.format(
                    images=self.dest,nmin=nmin,nmax=nmax)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)
    return RequiredLength


def resize_img(img_list, scale_percent):
    """
    resize the images in the img_list
    @param img_list: a list containing input images
    @param scale_percent: the scale percentage to which the images are scaled from the original images
    @return: a new list of resized images
    """
    output = []
    for img in img_list:
        # calculate the percent of original dimensions
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)

        # dsize
        dsize = (width, height)

        # resize image
        new_img = cv2.resize(img, dsize)

        output.append(new_img)

    return output


def compute_avgs_overlap(src_img, dst_img, offset_x, offset_y):
    """

    @param src_img: source image
    @param dst_img: target image
    @return: average RGB intensity of each image in overlapping region
    """
    overlap_src = []  # contains RGB values in the overlapping region
    overlap_dst = []

    w = dst_img.shape[1]
    h = dst_img.shape[0]
    for i in range(h):
        for j in range(w):
            if src_img[i + offset_y][j + offset_x].any() != 0 and dst_img[i][j].any() != 0:
                overlap_src.append(src_img[i + offset_y][j + offset_x])
                overlap_dst.append(dst_img[i][j])

    sum_rgb_src = np.zeros((1, 3))
    avg_rgb_src = np.zeros((1, 3))
    for rgb in overlap_src:
        sum_rgb_src += rgb

    avg_rgb_src = sum_rgb_src / len(overlap_src)

    sum_rgb_dst = np.zeros((1, 3))
    avg_rgb_dst = np.zeros((1, 3))
    for rgb in overlap_dst:
        sum_rgb_dst += rgb

    avg_rgb_dst = sum_rgb_dst / len(overlap_dst)

    return (avg_rgb_src, avg_rgb_dst)


def gain_compensation(src_img, dst_img, offset_x, offset_y):
    avg_rgb_src, avg_rgb_dst = compute_avgs_overlap(src_img, dst_img, offset_x, offset_y)

    reduce_dst = False
    if avg_rgb_src.sum() < avg_rgb_dst.sum():  # dst is lighter
        ratio_gain = avg_rgb_src / avg_rgb_dst
        reduce_dst = True
    else:
        ratio_gain = avg_rgb_dst / avg_rgb_src  # src is lighter

    if reduce_dst:
        # normalize target image
        dst_img = dst_img * ratio_gain
        # for i in range(dst_img.shape[0]):
        #     for j in range(dst_img.shape[1]):
        #         dst_img[i][j] = dst_img[i][j] * (ratio_gain)
    else:
        # normalize source image
        src_img = src_img * ratio_gain
        # for i in range(src_img.shape[0]):
        #     for j in range(src_img.shape[1]):
        #         src_img[i][j] = src_img[i][j] * (ratio_gain)

    return src_img.astype(np.uint8), dst_img.astype(np.uint8)


def alpha_blending(src_img, dst_img, offset_x, offset_y):
    """
    blend the overlapping region using alpha blending technique
    @param src_img: source image
    @param dst_img: dst(target) image
    @param offset_x: offset along x direction
    @param offset_y: offset along y direction
    @return: the blended image using alpha blending
    """
    w = dst_img.shape[1]
    h = dst_img.shape[0]

    for i in range(h):
        for j in range(w):
            if src_img[i+offset_y][j+offset_x].any()==0:  # add the translated right image
                src_img[i+offset_y][j+offset_x] = dst_img[i][j]
            elif dst_img[i][j].any() != 0:  # blend the overlapping region
                src_img[i + offset_y][j + offset_x] = alpha_blending_helper((i + offset_y, j + offset_x), src_img,
                                                                            dst_img, (offset_x, offset_y))

    return src_img


def alpha_blending_helper(p, src_img, dst_img, offsets):
    """
    helper function for alpha blending
    @param p: pixel location p
    @param src_img: source image
    @param dst_img: dst(target) image
    @param offsets: list containing offset_x and offset_y
    @return: pixel color(intensity) at p
    """
    # calculate image center of src and dst
    center_src = (src_img.shape[1]/2, src_img.shape[0]/2)
    center_dst = (dst_img.shape[1]/2 + offsets[1], dst_img.shape[0]/2 + offsets[0])

    # calculate distance between p and image centers
    dist_p_src = math.sqrt((p[0] - center_src[0]) ** 2 + (p[1] - center_src[1]) ** 2)
    dist_p_dst = math.sqrt((p[0] - center_dst[0]) ** 2 + (p[1] - center_dst[1]) ** 2)

    # assign alpha according to the calculated distance between p and image centers
    if (dist_p_src + dist_p_dst) == 0:
        alpha_src = 0.5
        alpha_dst = 0.5
    else:
        alpha_src = dist_p_dst / (dist_p_src + dist_p_dst)
        alpha_dst = dist_p_src / (dist_p_src + dist_p_dst)

    # compute color at pixel p
    src = alpha_src * src_img[p[0]][p[1]]
    dst = alpha_dst * dst_img[p[0] - offsets[1]][p[1] - offsets[0]]

    return (src + dst) / (alpha_src + alpha_dst)


def avg_blending(src_img, dst_img, offset_x, offset_y):
    """
    blend the overlapping region using avg blending
    @param src_img: source image
    @param dst_img: dst(target) image
    @param offset_x: offset along x direction
    @param offset_y: offset along y direction
    @return: blended image using average blending
    """
    w = dst_img.shape[1]
    h = dst_img.shape[0]

    for i in range(h):
        for j in range(w):
            if src_img[i+offset_y][j+offset_x].any()==0:  # add the translated right image
                src_img[i+offset_y][j+offset_x] = dst_img[i][j]
            elif dst_img[i][j].any() != 0:  # blend the overlapping region
                src_img[i+offset_y][j+offset_x] = src_img[i+offset_y][j+offset_x]/2 + dst_img[i][j]/2

    return src_img


def feathering(src_img, dst_img, offset_x, offset_y):
    """
    feathering blending method
    @param src_img: source image
    @param dst_img: target image
    @param offset_x: offset along x direction
    @param offset_y: offset along y direction
    @return: blended image using feathering blending
    """
    window_size = (src_img.shape[1] / 4) * 3  # this can be changed for reseach purposes

    w = dst_img.shape[1]
    h = dst_img.shape[0]
    for i in range(h):
        for j in range(w):
            if src_img[i+offset_y][j+offset_x].any()==0:  # add the translated right image
                src_img[i+offset_y][j+offset_x] = dst_img[i][j]
            elif dst_img[i][j].any() != 0:  # blend the overlapping region
                src_img[i + offset_y][j + offset_x] = feathering_helper(src_img[i + offset_y][j + offset_x], j,
                                                                   window_size, True)
                dst_img[i][j] = feathering_helper(dst_img[i][j], j, window_size, False)
                src_img[i + offset_y][j + offset_x] = src_img[i + offset_y][j + offset_x] + dst_img[i][j]

    return src_img


def feathering_helper(rgb_values, x, window_size, is_source=True):
    """
    feathering helper function
    @param rgb_values: RGB values for a pixel
    @param x: width(x) location of a pixel
    @param window_size: window size of feathering
    @param is_source: flag indicating source image
    @return: intensity value after feathering for a certain pixel
    """
    if is_source:
        mask = (1 / window_size) * (window_size - x)
    else:
        mask = (-1 / window_size) * (window_size - x) + 1

    return np.multiply(rgb_values, mask, casting="unsafe")


def adjust_mask(mask, rightmost_overlap_x, leftmost_overlap_x):
    """
    adjust the mask according to the overlapping region
    @param mask: pyramid mask for pyramid blending technique
    @param rightmost_overlap_x: rightmost x position in overlapping region
    @param leftmost_overlap_x: leftmost x position in overlapping region
    @return:
    """
    mid_x = (rightmost_overlap_x + leftmost_overlap_x) / 2

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if j == mid_x:
                mask[i][j] = np.array([.5, .5, .5])
            elif j > mid_x:
                mask[i][j] = np.array([0, 0, 0])

    return mask

def gaussian_pyramid(src_img, level):
    """
    gaussian pyramid method
    @param src_img: source image
    @param level: pyramid level
    @return: a gaussian pyramid for a particular input source image
    """
    g = src_img.copy()
    gaussian_pyramid = [g]
    for i in range(0, level-1):
        g = cv2.pyrDown(g)  # Gaussian blurring + down sampling
        gaussian_pyramid.append(np.float32(g))

    return gaussian_pyramid


def laplacian_pyramid(gaussian_pyramid, level):
    """
    form laplacian pyramid given the gaussian pyramid
    @param gaussian_pyramid: Gaussian pyramid
    @param level: laplacian pyramid level
    @return: corresponding Laplacian pyramid
    """
    lap_pyramid = [gaussian_pyramid[-1]]
    for i in range(level - 1, 0, -1):
        expand = cv2.resize(gaussian_pyramid[i], (gaussian_pyramid[i-1].shape[1], gaussian_pyramid[i-1].shape[0]))
        # expand = cv2.pyrUp(gaussian_pyramid[i])
        lap = np.subtract(gaussian_pyramid[i-1], expand)
        lap_pyramid.append(lap)

    return lap_pyramid


def form_new_laplacian_pyramid(L_src, L_dst, lp_mask):
    """
    form the new pyramid given two laplacian pyramids
    @param L_src: source image laplacian pyramid
    @param L_dst: target image laplacian pyramid
    @param lp_mask: laplacian mask
    @return: newly constructed laplacian pyramid
    """
    LS = []
    for l_src, l_dst, lp_m in zip(L_src, L_dst, lp_mask):
        ls = l_src * lp_m + l_dst * (1.0 - lp_m)
        LS.append(ls)

    return LS


def reconstruct(LS, level):
    """
    reconstruct blended image according to the new LS pyramid
    @param LS: newly constructed laplacian
    @param level: pyramid levels
    @return: reconstructed image
    """
    ls = LS[0]
    for i in range(1, level):
        expand = cv2.resize(ls, (LS[i].shape[1], LS[i].shape[0]))
        ls = np.add(expand, LS[i])

    return ls


def pyramid_blending(img_src, newM, img_final, img_dst, offset_x, offset_y, rightmost_overlap_x):
    """
    pyramid blending to create mosaic for the given two images and their mask
    @param img_src: original image for source image before warping
    @param newM: new homography after translating dst image
    @param img_final: src image after warping
    @param img_dst: dst image to be translated
    @param offset_x: offset along x direction
    @param offset_y: offset along y direction
    @return: image blended using pyramid method
    """

    m = np.ones_like(img_src, dtype='float32')
    mask = cv2.warpPerspective(m, newM, (img_final.shape[1], img_final.shape[0]))
    mask = adjust_mask(mask, rightmost_overlap_x, offset_x)

    w = img_dst.shape[1]
    h = img_dst.shape[0]

    # create a new image containing the translated dst image
    new_img_dst = np.zeros((img_final.shape[0], img_final.shape[1], 3), dtype='float32')
    for i in range(h):
        for j in range(w):
            if img_final[i + offset_y][j + offset_x].any() == 0:  # add the translated right image
                new_img_dst[i + offset_y][j + offset_x] = img_dst[i][j]
            elif img_dst[i][j].any() != 0:
                new_img_dst[i + offset_y][j + offset_x] = img_dst[i][j]

    # laplacian pyramid blending
    lpb = pyramid_blending_helper(img_final, new_img_dst, mask, 3)

    return lpb.astype(np.uint8)


def pyramid_blending_helper(src_img, dst_img, mask, level=6):
    """
    pyramid blending helper function
    @param src_img: source image
    @param dst_img: target image
    @param mask: mask image
    @param level: pyramid levels
    @return: the blended output image
    """

    # create gaussain pyramids for all input images
    gp_src = gaussian_pyramid(src_img, level)
    gp_dst = gaussian_pyramid(dst_img, level)
    gp_mask = gaussian_pyramid(mask, level)

    # create laplacian pyramids for all input images
    lp_src = laplacian_pyramid(gp_src, level)
    lp_dst = laplacian_pyramid(gp_dst, level)
    lp_mask = [gp_mask[-1]]

    for i in range(level-1, 0, -1):  # reverse the masks
        lp_mask.append(gp_mask[i-1])

    # blend images according to mask in each level -> form new pyramid
    LS = form_new_laplacian_pyramid(lp_src, lp_dst, lp_mask)

    # reconstruct image according to the new laplacian pyramid
    return reconstruct(LS, level)


def vertices_location(img):
    """
    find the four vertices locations given an image
    @param img: input image
    @return: a four-element tuple indicating the vertices locations (up_left, up_right, down_right, down_left)
    """
    up_left = np.array([0, 0, 1])
    up_right = np.array([img.shape[1], 0, 1])
    down_right = np.array([img.shape[1], img.shape[0], 1])
    down_left = np.array([0, img.shape[0], 1])

    return (up_left, up_right, down_right, down_left)


def transform_vertices(vertices, homography):
    """
    transform the vertices given the homography
    @param vertices: original vertices locations
    @param homography: homography for transformation
    @return: transformed vertices locations
    """
    output = []
    for v in list(vertices):
        new_v = np.dot(homography, v)
        new_v = new_v / new_v[-1]
        output.append(new_v)

    return tuple(output)

def find_stitch_order(img_list):
    """
    find the correct order to stitch the images
    @param img_list: list containing the input images to be stitched in random order
    @return: a new list containing images to be stitched in the correct order
    """
    # initilize SIFT to find feature points and descriptor in an image
    sift = cv2.xfeatures2d.SIFT_create()

    imgs_keyPoints = {}
    imgs_des = {}
    for i in range(0, len(img_list)):
        kp, des = sift.detectAndCompute(img_list[i], None)
        imgs_keyPoints[i] = kp
        imgs_des[i] = des

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    inlier_matrix = np.zeros((len(img_list), len(img_list)))

    for i in range(0, len(img_list)):
        for j in range(0, len(img_list)):
            if i == j: continue
            matches = flann.knnMatch(imgs_des[i], imgs_des[j], k=2)  # find matching points between two images i and j

            # store all the good matches as per Lowe's ratio test.
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            num_inliers = len(good)
            if num_inliers > 8 + 0.3 * len(matches):
                inlier_matrix[i][j] = num_inliers
            else:
                inlier_matrix[i][j] = 0

    idx_total_inliers = {}  # index -> total inliers for each image index
    for i in range(0, len(img_list)):
        idx_total_inliers[i] = inlier_matrix[i].sum()

    new_list_idx = {k: v for k, v in sorted(idx_total_inliers.items(), key=lambda item: item[1])}.keys()
    output = []
    for i in new_list_idx:
        output.append(img_list[i])

    return output
