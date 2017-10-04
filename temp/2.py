import math
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------------------------------
def feature_matching(img1, img2, savefig=False):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches2to1 = flann.knnMatch(des2,des1,k=2)

    matchesMask_ratio = [[0,0] for i in xrange(len(matches2to1))]
    match_dict = {}
    for i,(m,n) in enumerate(matches2to1):
        if m.distance < 0.7*n.distance:
            matchesMask_ratio[i]=[1,0]
            match_dict[m.trainIdx] = m.queryIdx

    good = []
    recip_matches = flann.knnMatch(des1,des2,k=2)
    matchesMask_ratio_recip = [[0,0] for i in xrange(len(recip_matches))]

    for i,(m,n) in enumerate(recip_matches):
        if m.distance < 0.7*n.distance: # ratio
            if m.queryIdx in match_dict and match_dict[m.queryIdx] == m.trainIdx: #reciprocal
                good.append(m)
                matchesMask_ratio_recip[i]=[1,0]



    if savefig:
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask_ratio_recip,
                           flags = 0)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,recip_matches,None,**draw_params)

        plt.figure(),plt.xticks([]),plt.yticks([])
        plt.imshow(img3,)
        plt.savefig("feature_matching.png",bbox_inches='tight')

    return ([ kp1[m.queryIdx].pt for m in good ],[ kp2[m.trainIdx].pt for m in good ])

'''
Warp an image from cartesian coordinates (x, y) into cylindrical coordinates (theta, h)
Returns: (image, mask)
Mask is [0,255], and has 255s wherever the cylindrical images has a valid value.
Masks are useful for stitching

Usage example:

    im = cv2.imread("myimage.jpg",0) #grayscale
    h,w = im.shape
    f = 700
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
    imcyl = cylindricalWarpImage(im, K)
'''
def cylindricalWarpImage(img1, K, savefig=False):
    f = K[0,0]

    im_h,im_w = img1.shape

    # go inverse from cylindrical coord to the image
    # (this way there are no gaps)
    cyl = np.zeros_like(img1)
    cyl_mask = np.zeros_like(img1)
    cyl_h,cyl_w = cyl.shape
    x_c = float(cyl_w) / 2.0
    y_c = float(cyl_h) / 2.0
    for x_cyl in np.arange(0,cyl_w):
        for y_cyl in np.arange(0,cyl_h):
            theta = (x_cyl - x_c) / f
            h     = (y_cyl - y_c) / f

            X = np.array([math.sin(theta), h, math.cos(theta)])
            X = np.dot(K,X)
            x_im = X[0] / X[2]
            if x_im < 0 or x_im >= im_w:
                continue

            y_im = X[1] / X[2]
            if y_im < 0 or y_im >= im_h:
                continue

            cyl[int(y_cyl),int(x_cyl)] = img1[int(y_im),int(x_im)]
            cyl_mask[int(y_cyl),int(x_cyl)] = 255


    if savefig:
        plt.imshow(cyl, cmap='gray')
        plt.savefig("cyl.png",bbox_inches='tight')

    return (cyl,cyl_mask)

'''
Calculate the geometric transform (only affine or homography) between two images,
based on feature matching and alignment with a robust estimator (RANSAC).

Returns: (M, pts1, pts2, mask)
Where: M    is the 3x3 transform matrix
       pts1 are the matched feature points in image 1
       pts2 are the matched feature points in image 2
       mask is a binary mask over the lists of points that selects the transformation inliers

Usage example:
    im1 = cv2.imread("image1.jpg", 0)
    im2 = cv2.imread("image2.jpg", 0)
    (M, pts1, pts2, mask) = getTransform(im1, im2)

    # for example: transform im1 to im2's plane
    # first, make some room around im2
    im2 = cv2.copyMakeBorder(im2,200,200,500,500, cv2.BORDER_CONSTANT)
    # then transform im1 with the 3x3 transformation matrix
    out = cv2.warpPerspective(im1, M, (im1.shape[1],im2.shape[0]), dst=im2.copy(), borderMode=cv2.BORDER_TRANSPARENT)

    plt.imshow(out, cmap='gray')
    plt.show()
'''
def getTransform(src, dst, method='affine'):
    pts1,pts2 = feature_matching(src,dst)

    src_pts = np.float32(pts1).reshape(-1,1,2)
    dst_pts = np.float32(pts2).reshape(-1,1,2)

    if method == 'affine':
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=1.0)
        # M = np.append(M, [[0,0,1]], axis=0)

    if method == 'homography':
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

    return (M, pts1, pts2, mask)

#---------------------------------------------------------------
def RMSD(questionID, target, master):
    # Get width, height, and number of channels of the master image
    master_height, master_width = master.shape[:2]
    master_channel = len(master.shape)

    # Get width, height, and number of channels of the target image
    target_height, target_width = target.shape[:2]
    target_channel = len(target.shape)

    # Validate the height, width and channels of the input image
    if (master_height != target_height or master_width != target_width
            or master_channel != target_channel):
        return -1
    else:
        nonZero_target = cv2.countNonZero(target)
        nonZero_master = cv2.countNonZero(master)

        if (questionID == 1):
            if (nonZero_target < 1200000):
                return -1
        elif (questionID == 2):
            if (nonZero_target < 700000):
                return -1
        else:
            return -1

        total_diff = 0.0
        master_channels = cv2.split(master)
        target_channels = cv2.split(target)

        for i in range(0, len(master_channels), 1):
            dst = cv2.absdiff(master_channels[i], target_channels[i])
            dst = cv2.pow(dst, 2)
            mean = cv2.mean(dst)
            total_diff = total_diff + mean[0]**(1 / 2.0)

        return total_diff
#------------------------------------------------
f = 450
print f

imcc = cv2.imread("../hw2/input1.png", 0)
h,w = imcc.shape
K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
imcc, maskcc = cylindricalWarpImage(imcc, K)



imrc = cv2.imread("../hw2/input2.png", 0)
h,w = imrc.shape
K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
imrc, maskrc = cylindricalWarpImage(imrc, K)


imlc = cv2.imread("../hw2/input3.png", 0)
h,w = imlc.shape
K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
imlc, masklc = cylindricalWarpImage(imlc, K)


imcc = cv2.copyMakeBorder(imcc,50,50,300,300, cv2.BORDER_CONSTANT)
imrc = cv2.copyMakeBorder(imrc,50,50,300,300, cv2.BORDER_CONSTANT)
imlc = cv2.copyMakeBorder(imlc,50,50,300,300, cv2.BORDER_CONSTANT)


(M, pts1, pts2, mask) = getTransform(imrc, imcc)
outrc = cv2.warpAffine(imrc, M, (imcc.shape[1],imcc.shape[0]))

(M, pts1, pts2, mask) = getTransform(imlc, imcc)
outlc = cv2.warpAffine(imlc, M, (imcc.shape[1],imcc.shape[0]))



cv2.namedWindow("imw",cv2.WINDOW_NORMAL)
'''
cv2.imshow("imw", outrc)
cv2.waitKey()
cv2.imshow("imw", imcc)
cv2.waitKey()
cv2.imshow("imw", outlc)
cv2.waitKey()
'''

imfinal = np.maximum.reduce([outrc, imcc, outlc])
cv2.imshow("imw", imfinal)
cv2.waitKey()

imdemo = cv2.imread("../hw2/example_output2.png",0)
cv2.imshow("imw", imdemo)
cv2.waitKey()

print RMSD(2, imdemo, imfinal)