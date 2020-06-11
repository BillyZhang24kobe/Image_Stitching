# image stitching main function
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import *


def image_stitching(img_list, blending, gain_comp, drawInliers):
    """
    main function for image stitching
    @param img_list: list of images to be stitched
    @param blending: blending technique selected
    @param gain_compensation: flag indicating whether to adopt gain compensation
    @param drawInliers: flag for drawInliers
    @return: the final stitched image
    """
    img_src = img_list[0]

    for i in range(1, len(img_list)):
        img_dst = img_list[i]

        # point locations for the four vertices of source image
        src_vertices = vertices_location(img_src)

        ###################################################
        # Step 1: Detect SIFT points for each of the images
        ###################################################
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img_src, None)
        kp2, des2 = sift.detectAndCompute(img_dst, None)

        ###############################################################
        # Step 2: Find matching points among those found in two images.
        ###############################################################
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)  # find points that are close to each other

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        ##################################################
        # Step 3: Estimate Homography based on the matches
        ##################################################
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Find homography using RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


        ####################################################
        # Step 4: Compute the offset along x and y direction
        ####################################################
        # vertices location for src image after homography
        src_vertices_transform = transform_vertices(src_vertices, M)

        # determine the x offset for the final output image based on the dst image
        src_up_left_transform = src_vertices_transform[0]
        src_down_left_transform = src_vertices_transform[3]
        if min(src_up_left_transform[0], src_down_left_transform[0]) < 0:
            offset_x = int((-1) * min(src_up_left_transform[0], src_down_left_transform[0]))
        else:
            offset_x = 0

        # determine the y offset and y_expand for the final output image based on the dst image
        src_up_right_transform = src_vertices_transform[1]
        src_down_right_transform = src_vertices_transform[2]
        offset_y = 0
        y_expand = 0
        if min(src_up_left_transform[1], src_up_right_transform[1]) < 0:
            offset_y = int((-1) * min(src_up_left_transform[1], src_up_right_transform[1]))

        if max(src_down_right_transform[1], src_down_left_transform[1]) > img_dst.shape[0]:
            y_expand = int(max(src_down_right_transform[1], src_down_left_transform[1]) - img_dst.shape[0])

        ################################################
        # Step 5: draw the inliers if specified by users
        ################################################
        # visualize inliers if drawInliers is true
        if drawInliers:
            matchesMask = mask.ravel().tolist()

            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)

            img3 = cv2.drawMatches(img_src, kp1, img_dst, kp2, good, None, **draw_params)
            plt.figure()
            plt.imshow(img3)
            plt.xticks([]), plt.yticks([])

        ##########################################################
        # Step 6: for each matched points of the right image,
        # add the translation from right image to the final image.
        ##########################################################

        w = img_dst.shape[1]
        h = img_dst.shape[0]

        dst_pts1 = dst_pts.copy()  # matched points on the right image
        dst_pts1[:, :, 0] = dst_pts1[:, :, 0] + offset_x
        dst_pts1[:, :, 1] = dst_pts1[:, :, 1] + offset_y

        ###########################################
        # Step 7: Compute new homography using the
        # new set of coordinates of the right image
        ###########################################
        newM, mask = cv2.findHomography(src_pts, dst_pts1, cv2.RANSAC, 5.0)

        src_vertices_transform = transform_vertices(src_vertices, newM)
        src_up_right_transform = src_vertices_transform[1]
        src_down_right_transform = src_vertices_transform[2]
        #############################
        # Step 8: Warp the left image
        #############################
        img_final = cv2.warpPerspective(img_src, newM, (w + offset_x, h + offset_y + y_expand))

        #######################################################
        # Step 9: Adopt gain compensation if requested by users
        #######################################################
        if gain_comp:
            img_final, img_dst = gain_compensation(img_final, img_dst, offset_x, offset_y)

        ############################################################
        # Step 10: Add the translated right image to the final image
        # and blend the overlapping region according to the selected
        # blending method
        ############################################################
        if blending == 'avg':  # average blending
            img_src = avg_blending(img_final, img_dst, offset_x, offset_y)
        elif blending == 'alpha':  # alpha blending
            img_src = alpha_blending(img_final, img_dst, offset_x, offset_y)
        elif blending == 'feathering':  # feathering blending
            img_src = feathering(img_final, img_dst, offset_x, offset_y)
        elif blending == 'pyramid':  # pyramid blending
            img_src = pyramid_blending(img_src, newM, img_final, img_dst, offset_x, offset_y,
                                       max(src_up_right_transform[0], src_down_right_transform[0]))

    return img_src.astype(np.uint8)