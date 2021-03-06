import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

# import skvideo.io

face_cascade = cv2.CascadeClassifier(
    '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')


def help_message():
    print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
    print("[Question Number]")
    print("1 Camshift")
    print("2 Particle Filter")
    print("3 Kalman Filter")
    print("4 Optical Flow")
    print("[Input_Video]")
    print("Path to the input video")
    print("[Output_Directory]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "02-1.avi " + "./")


def detect_one_face(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0, 0, 0, 0)
    return faces[0]


def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c, r, w, h = window
    roi = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi,
                       np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist


def particleevaluator(back_proj, particle):
    pass
    return back_proj[particle[1],particle[0]]


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i + 1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0 + i) / n for i in range(n)]:
        while u > C[j]:
            j += 1
        indices.append(j - 1)
    return indices

def skeleton_tracker_optical(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name, "w")

    frameCounter = 0
    # read first frame
    ret, frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c, r, w, h = detect_one_face(frame)
    # print c, r, w, h

    pt_x = c + (w / 2)
    pt_y = r + (h / 2)
    pt = (0, pt_x, pt_y)

    # Write track point for first frame
    output.write("%d,%d,%d\n" % pt)  # Write as 0,pt_x,pt_y

    # set the initial tracking window
    track_window = (c, r, w, h)

    # initialize the tracker
    # e.g. kf = cv2.KalmanFilter(4,2,0)
    # or: particles = np.ones((n_particles, 2), int) * initial_pos
    state = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')  # initial position
    statePre = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')
    kalman = cv2.KalmanFilter(4, 2, 0)
    #4 = dimensionality of state
    #2 = dimensionality of measurement
    #0 = dimensionality of control vector
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                        [0., 1., 0., .1],
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4) #5
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2) #3
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state

    while (1):
        ret, frame = v.read()  # read another frame
        frameCounter = frameCounter + 1
        if ret == False:
            break
        c1, r1, w1, h1 = detect_one_face(frame)
        center = np.array([c1 + w1/2, r1 + h1/2])
        prediction = kalman.predict().reshape(4,)



        pt_x = c1 + (w1 / 2)
        pt_y = r1 + (h1 / 2)
        pt = (frameCounter, pt_x, pt_y)

        measurement = [pt_x, pt_y]

        if h1>0:
            kalman.correct(tuple(measurement))
            # img2 = cv2.circle(frame, (pt_x, pt_y), 2, 255, -1)

        else:
            pt_x = int(prediction[0])
            pt_y = int(prediction[1])
            # img2 = cv2.circle(frame, (pt_x, pt_y), 4, 25, -1)

        # cv2.imshow("i", img2)
        pt = (frameCounter, pt_x, pt_y)
        # print pt
        # k = cv2.waitKey(60) & 0xff
        # if k == 27:
            # break

        # write the result to the output file
        output.write("%d,%d,%d\n" % pt)  # Write as frame_index,pt_x,pt_y

    output.close()


def skeleton_tracker_kalman(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name, "w")

    frameCounter = 0
    # read first frame
    ret, frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c, r, w, h = detect_one_face(frame)
    # print c, r, w, h

    pt_x = c + (w / 2)
    pt_y = r + (h / 2)
    pt = (0, pt_x, pt_y)

    # Write track point for first frame
    output.write("%d,%d,%d\n" % pt)  # Write as 0,pt_x,pt_y

    # set the initial tracking window
    track_window = (c, r, w, h)

    # initialize the tracker
    # e.g. kf = cv2.KalmanFilter(4,2,0)
    # or: particles = np.ones((n_particles, 2), int) * initial_pos
    state = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')  # initial position
    statePre = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')
    kalman = cv2.KalmanFilter(4, 2, 0)
    #4 = dimensionality of state
    #2 = dimensionality of measurement
    #0 = dimensionality of control vector
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                        [0., 1., 0., .1],
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4) #5
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2) #3
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state

    while (1):
        ret, frame = v.read()  # read another frame
        frameCounter = frameCounter + 1
        if ret == False:
            break
        c1, r1, w1, h1 = detect_one_face(frame)
        center = np.array([c1 + w1/2, r1 + h1/2])
        prediction = kalman.predict().reshape(4,)



        pt_x = c1 + (w1 / 2)
        pt_y = r1 + (h1 / 2)
        pt = (frameCounter, pt_x, pt_y)

        measurement = [pt_x, pt_y]

        if h1>0:
            kalman.correct(tuple(measurement))
            # img2 = cv2.circle(frame, (pt_x, pt_y), 2, 255, -1)

        else:
            pt_x = int(prediction[0])
            pt_y = int(prediction[1])
            # img2 = cv2.circle(frame, (pt_x, pt_y), 4, 25, -1)

        # cv2.imshow("i", img2)
        pt = (frameCounter, pt_x, pt_y)
        # print pt
        # k = cv2.waitKey(60) & 0xff
        # if k == 27:
            # break

        # write the result to the output file
        output.write("%d,%d,%d\n" % pt)  # Write as frame_index,pt_x,pt_y

    output.close()


def skeleton_tracker_particle(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name, "w")

    frameCounter = 0
    # read first frame
    ret, frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c, r, w, h = detect_one_face(frame)

    pt_x = c + (w / 2)
    pt_y = r + (h / 2)
    pt = (frameCounter, pt_x, pt_y)

    # Write track point for first frame
    output.write("%d,%d,%d\n" % pt)  # Write as 0,pt_x,pt_y

    # set the initial tracking window
    track_window = (c, r, w, h)

    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c, r, w,
                                                h))  # this is provided for you

    # initialize the tracker
    # e.g. kf = cv2.KalmanFilter(4,2,0)
    # or: particles = np.ones((n_particles, 2), int) * initial_pos
    n_particles = 200

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)


    init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
    particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
    f0 = particleevaluator(dst, particles.T) * np.ones(n_particles) # Evaluate appearance model
    weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)
    stepsize = 20

    while (1):
        ret, frame = v.read()  # read another frame
        if ret == False:
            break
        frameCounter = frameCounter + 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        im_w=frame.shape[1]
        im_h=frame.shape[0]
        np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")
        particles = particles.clip(np.zeros(2), np.array((im_w,im_h))-1).astype(int)
        f = particleevaluator(dst, particles.T) # Evaluate particles
         # update particles
        weights = np.float32(f.clip(1))             # Weight ~ histogram response
        weights /= np.sum(weights)                  # Normalize w
        mean = np.sum(particles.T * weights, axis=1).astype(int)

        if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
            particles = particles[resample(weights),:]  # Resample particles according to weights

        # write the result to the output file
        pt_to_write = (frameCounter, mean[0], mean[1])
        output.write("%d,%d,%d\n" % pt_to_write)  # Write as frame_index,pt_x,pt_y



        # img2 = cv2.circle(frame, (mean[0], mean[1]), 2, 255, -1)
        # cv2.imshow("i", img2)
        # print pt_to_write
        # k = cv2.waitKey(60) & 0xff
        # if k == 27:
            # break

    output.close()


def skeleton_tracker_camshift(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name, "w")

    frameCounter = 0
    # read first frame
    ret, frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c, r, w, h = detect_one_face(frame)

    pt_x = c + (w / 2)
    pt_y = r + (h / 2)
    pt = (frameCounter, pt_x, pt_y)

    # Write track point for first frame
    output.write("%d,%d,%d\n" % pt)  # Write as 0,pt_x,pt_y

    # set the initial tracking window
    track_window = (c, r, w, h)

    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c, r, w,
                                                h))  # this is provided for you

    # initialize the tracker
    # e.g. kf = cv2.KalmanFilter(4,2,0)
    # or: particles = np.ones((n_particles, 2), int) * initial_pos
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1)

    while (1):
        ret, frame = v.read()  # read another frame
        if ret == False:
            break
        frameCounter = frameCounter + 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)

        mean_x = (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4
        mean_y = (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4

        pt_to_write = (frameCounter, mean_x, mean_y)
        # write the result to the output file
        output.write(
            "%d,%d,%d\n" % pt_to_write)  # Write as frame_index,pt_x,pt_y

    output.close()


if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    # video = skvideo.io.VideoCapture(sys.argv[2])
    video = cv2.VideoCapture(sys.argv[2])

    if (question_number == 1):
        skeleton_tracker_camshift(video, "output_camshift.txt")
    elif (question_number == 2):
        skeleton_tracker_particle(video, "output_particle.txt")
    elif (question_number == 3):
        skeleton_tracker_kalman(video, "output_kalman.txt")
    elif (question_number == 4):
        skeleton_tracker_optical(video, "output_of.txt")
'''
For Kalman Filter:

# --- init

state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                    [0., 1., 0., .1],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
kalman.measurementMatrix = 1. * np.eye(2, 4)
kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
kalman.errorCovPost = 1e-1 * np.eye(4, 4)
kalman.statePost = state


# --- tracking

prediction = kalman.predict()

# ...
# obtain measurement

if measurement_valid: # e.g. face found
    # ...
    posterior = kalman.correct(measurement)

# use prediction or posterior as your tracking result
'''
'''
For Particle Filter:

# --- init

# a function that, given a particle position, will return the particle's "fitness"
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]

# hist_bp: obtain using cv2.calcBackProject and the HSV histogram
# c,r,w,h: obtain using detect_one_face()
n_particles = 200

init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
f0 = particleevaluator(hist_bp, pos) * np.ones(n_particles) # Evaluate appearance model
weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)


# --- tracking

# Particle motion model: uniform step (TODO: find a better motion model)
np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")

# Clip out-of-bounds particles
particles = particles.clip(np.zeros(2), np.array((im_w,im_h))-1).astype(int)

f = particleevaluator(hist_bp, particles.T) # Evaluate particles
weights = np.float32(f.clip(1))             # Weight ~ histogram response
weights /= np.sum(weights)                  # Normalize w
pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average

if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
    particles = particles[resample(weights),:]  # Resample particles according to weights
# resample() function is provided for you
'''
