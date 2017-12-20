# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================

import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle
import sys

def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")


def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(
        cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0),
        fx=scale_factor,
        fy=scale_factor)
    ref_black = cv2.resize(
        cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0),
        fx=scale_factor,
        fy=scale_factor)
    rgbimg = cv2.resize(cv2.imread("images/pattern001.jpg"), (0,0), fx=scale_factor,fy=scale_factor)

    ref_avg = (ref_white + ref_black) / 2.0

    ref_on = ref_avg + 0.0  # a threshold for ON pixels
    ref_off = ref_avg - 0.5  # add a small buffer region

    h, w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h, w), dtype=np.uint16)

    # analyze the binary patterns from the camera
    for i in range(0, 15):
        # read the file
        # patt_gray = cv2.resize(
        #     cv2.imread("images/pattern%03d.jpg" % (i + 2), cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0),
        #     fx=scale_factor,
        #     fy=scale_factor)

        patt_gray = cv2.resize(
            cv2.imread("images/pattern%03d.jpg" % (i + 2), cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0),
            fx=scale_factor,
            fy=scale_factor)

        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask

        # this code corresponds with the binary pattern code

        bit_code = np.uint16(1 << i)
        scan_bits = np.add(scan_bits, np.multiply(bit_code,on_mask))
        # print np.unique(scan_bits),"z\nz"

        # TODO: populate scan_bits by putting the bit_code according to on_mask


        #cv2.imshow("onm", 255*np.uint8(on_mask))
        #cv2.waitKey()

    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl", "r") as f:
        binary_codes_ids_codebook = pickle.load(f)

    rgbvals = []
    camera_points = []#np.array([[0,0]])
    projector_points = []#np.array([[0,0]])
    bin_image = np.zeros((scan_bits.shape[0], scan_bits.shape[1], 3))

    for x in range(w):
        for y in range(h):
            if not proj_mask[y, x]:
                continue  # no projection here
            if scan_bits[y, x] not in binary_codes_ids_codebook:
                continue  # bad binary code

            cp=[x/2.,y/2.]
            # camera_points.append(tuple(cp))
            # camera_points = np.concatenate((camera_points,np.array([[x/2, y/2]])), axis=0)
            # print camera_points.shape

            p_x, p_y = binary_codes_ids_codebook[scan_bits[y,x]]
            if p_x >= 1279 or p_y >= 799: # filter
                continue
            camera_points.append(cp)
            projector_points.append([p_x, p_y])
            # projector_points = np.concatenate((projector_points,np.array([[p_x, p_y]])))
            # print projector_points
            # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # TODO: find for the camera (x,y) the projector (p_x, p_y).
            # TODO: store your points in camera_points and projector_points
            bin_image[y,x] = np.array([0, p_y, p_x])
            rgbvals.append(rgbimg[y,x][::])
            # print "tgbvals", rgbimg[y,x][::]
            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2
    '''
    for i in range(len(projector_points)):
        print camera_points[i],":", projector_points[i]
    raw_input()
    '''
    rgbvals = np.array(rgbvals)
    bin_image[:,:,1] = bin_image[:,:,1]*255/800.0
    bin_image[:,:,2] = bin_image[:,:,2]*255/1280.0
    cv2.imwrite(sys.argv[1]+"correspondence.jpg", bin_image)


    # now that we have 2D-2D correspondances, we can triangulate 3D points!

    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl", "r") as f:
        d = pickle.load(f)
        camera_K = d['camera_K']
        camera_d = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

        # TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
        # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d
        # print cp, camera_K, camera_d
        camp=np.array(camera_points, dtype=np.float32)
        camp = camp.reshape((camp.shape[0], 1, 2))

        projp=np.array(projector_points, dtype=np.float32)
        projp = projp.reshape((projp.shape[0], 1, 2))

        norm_cp = cv2.undistortPoints(camp, camera_K, camera_d)
        norm_pp = cv2.undistortPoints(projp, projector_K, projector_d)

        undist_cp = norm_cp.reshape(norm_cp.shape[0], 2)
        undist_pp = norm_pp.reshape(norm_pp.shape[0], 2)

        # print norm_cp[1]

        # raw_input()

        # TODO: use cv2.triangulatePoints to triangulate the normalized points

        camera_R = np.eye(3)
        camera_t = np.zeros((3,1))
        camera_P = np.hstack((camera_R, camera_t))

        projector_P = np.hstack((projector_R, projector_t))

        print camera_P.shape, projector_P.shape

        points_3d = cv2.triangulatePoints(camera_P, projector_P, np.transpose(undist_cp), np.transpose(undist_pp))
        points_3d = cv2.convertPointsFromHomogeneous(np.transpose(points_3d))

        mask = (points_3d[:,:,2] > 200) & (points_3d[:,:,2] < 1400)
        mask = mask.reshape((mask.shape[0],))

        undist_cp = undist_cp[mask]
        undist_pp = undist_pp[mask]
        points_3d = points_3d[mask]
        rgbvals = rgbvals[mask]

        # TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
        # TODO: name the resulted 3D points as "points_3d"

        return points_3d, rgbvals


def write_3d_points(points_3d, rgbvals):
    # ===== DO NOT CHANGE THIS FUNCTION =====
    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name, "w") as f:
        for p in points_3d:
            f.write("%d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2]))

    output_name_new = sys.argv[1] + "output_color.xyz"
    with open(output_name_new, "w") as f:
        for p, rgb in zip(points_3d, rgbvals):
            f.write("%d %d %d %d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2], rgb[0], rgb[1], rgb[2]))

    # return points_3d, camera_points, projector_points


if __name__ == '__main__':
    # ===== DO NOT CHANGE THIS FUNCTION =====
    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d, rgbvals = reconstruct_from_binary_patterns()
    write_3d_points(points_3d, rgbvals)
