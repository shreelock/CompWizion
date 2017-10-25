import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
import maxflow
from scipy.spatial import Delaunay
import sys

drawing = False # true if mouse is pressed
draw_fg = True # if True, draw rectangle. Press 'm' to toggle to curve
fg_done = False
bg_done = False
ix,iy = -1,-1
brush_sz = 3

# mouse callback function
def draw_on_image(event,x,y,flags,param):
    global ix,iy,drawing,draw_fg, fg_done, bg_done, brush_sz

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if draw_fg == True:
                fg_done = True
                for i in range(-brush_sz, brush_sz+1):
                    for j in range(-brush_sz, brush_sz+1):
                        img_marking[y+i][x+j] = (0,0,255)
            else:
                bg_done = True
                for i in range(-brush_sz, brush_sz+1):
                    for j in range(-brush_sz, brush_sz+1):
                        img_marking[y+i][x+j] = (255,0,0)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if fg_done and bg_done:
            print "Segmentation can start"


if __name__ == '__main__':

    img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    img_marking = np.ones(img.shape, np.uint8)*255
    cv2.namedWindow('image')
    cv2.namedWindow('image_marking')

    cv2.setMouseCallback('image',draw_on_image)


    while(1):
        cv2.imshow('image',img)
        cv2.imshow('image_marking',img_marking)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('z'):
            draw_fg = not draw_fg
        elif k == 27:
            break
    cv2.destroyAllWindows()
