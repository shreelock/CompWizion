
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
        # plt.show()

    return ([ kp1[m.queryIdx].pt for m in good ],[ kp2[m.trainIdx].pt for m in good ])


def getTransform(src, dst, method='affine'):
    pts1,pts2 = feature_matching(src,dst)

    src_pts = np.float32(pts1).reshape(-1,1,2)
    dst_pts = np.float32(pts2).reshape(-1,1,2)

    if method == 'affine':
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
        M = np.append(M, [[0,0,1]], axis=0)

    if method == 'homography':
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

    return (M, pts1, pts2, mask)

imc = cv2.imread("../hw2/input1.png", 0)
imc = cv2.copyMakeBorder(imc,200,200,500,500, cv2.BORDER_CONSTANT)
imr = cv2.imread("../hw2/input2.png", 0)
iml = cv2.imread("../hw2/input3.png", 0)

(M, pts1, pts2, mask) = getTransform(imr, imc, "homography")
outcr = cv2.warpPerspective(imr, M, (imc.shape[1],imc.shape[0]))

(M, pts1, pts2, mask) = getTransform(iml, imc, "homography")
outlc = cv2.warpPerspective(iml, M, (imc.shape[1],imc.shape[0]))

cv2.namedWindow("imw",cv2.WINDOW_NORMAL)
cv2.imshow("imw", outcr)
cv2.waitKey()
cv2.imshow("imw", imc)
cv2.waitKey()
cv2.imshow("imw", outlc)
cv2.waitKey()

imfinal = np.maximum.reduce([outcr, imc, outlc])
cv2.imshow("imw", imfinal)
cv2.waitKey()

imdemo = cv2.imread("../hw2/example_output1.png",0)
cv2.imshow("imw", imdemo)
cv2.waitKey()
