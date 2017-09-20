# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Histogram equalization")
   print("2 Frequency domain filtering")
   print("3 Laplacian pyramid blending")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")                                # Single input, single output
   print(sys.argv[0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, three outputs
   print(sys.argv[0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, single output

# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================

def histogram_equalization(img_in):

   # Write histogram equalization here
   img_out = img_in # Histogram equalization result

   #####
   processed=np.zeros((img_in.shape[0],img_in.shape[1], 3)).astype("uint8")
   for i in xrange(3):
       b=img_in[:,:,i]

       hist = cv2.calcHist([b], [0], None,[256], [0,255])
       cdf = np.cumsum(hist)

       totalpixels = sum(hist) - 1
       min_of_cdf = min(cdf)
       equalizeFunc = np.vectorize(
           lambda p : round(
               ( (cdf[p]- min_of_cdf) * 255)
                / totalpixels )
           )


       d = equalizeFunc(b)
       np.clip(d, 0,255, out=d)
       processed[:,:,i] = d
   #####
   img_out = processed

   return True, img_out

def Question1():

   # Read in input images
   input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);

   # Histogram equalization
   succeed, output_image = histogram_equalization(input_image)

   # Write out the result
   output_name = sys.argv[3] + "output1.png"
   cv2.imwrite(output_name, output_image)

   return True

# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================

def ft(im, newsize=None):
   dft = np.fft.fft2(np.float32(im),newsize)
   return np.fft.fftshift(dft)

def ift(shift):
   f_ishift = np.fft.ifftshift(shift)
   img_back = np.fft.ifft2(f_ishift)
   return np.abs(img_back)


def low_pass_filter(img_in):

   # Write low pass filter here
   img_out = img_in # Low pass filter result
   ###
   lowpassfilter = np.uint8(np.zeros(img_in.shape))
   lowpassfilter[110:130, 150:170] = 1

   fftimg = ft(img_in)

   lpfftimg = fftimg*lowpassfilter
   lpimg = ift(lpfftimg)
   ###
   img_out = np.uint8(lpimg)
   return True, img_out

def high_pass_filter(img_in):
   # Write high pass filter here
   img_out = img_in # High pass filter result
   ###
   highpassfilter = np.uint8(np.ones(img_in.shape))
   highpassfilter[110:130, 150:170] = 0

   fftimg = ft(img_in)

   hpfftimg = fftimg*highpassfilter
   hpimg = ift(hpfftimg)
   ###
   img_out = np.uint8(hpimg)
   return True, img_out

def deconvolution(img_in):

   # Write deconvolution codes here
   img_out = img_in # Deconvolution result
   ###
   fftimg2 = ft(img_in)

   gaussianKernel = cv2.getGaussianKernel(21,5)
   gaussianKernel = gaussianKernel * gaussianKernel.T

   fftGk = ft(gaussianKernel, img_in.shape)

   deconvolve = fftimg2/fftGk

   unblurred_img = ift(deconvolve)
   ###
   img_out = unblurred_img
   return True, img_out

def Question2():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], 0);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH);

   # Low and high pass filter
   succeed1, output_image1 = low_pass_filter(input_image1)
   succeed2, output_image2 = high_pass_filter(input_image1)

   # Deconvolution
   succeed3, output_image3 = deconvolution(input_image2)

   # Write out the result
   output_name1 = sys.argv[4] + "output2LPF.png"
   output_name2 = sys.argv[4] + "output2HPF.png"
   output_name3 = sys.argv[4] + "output2deconv.png"
   cv2.imwrite(output_name1, output_image1)
   cv2.imwrite(output_name2, output_image2)
   cv2.imwrite(output_name3, cv2.convertScaleAbs(output_image3, alpha=255.0))

   return True

# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================

def laplacian_pyramid_blending(img_in1, img_in2):

   # Write laplacian pyramid blending codes here
   img_out = img_in1 # Blending result
   ###
   NO_OF_LEVELS = 5

   img_in1 = img_in1[:,:img_in1.shape[0]]
   img_in2 = img_in2[:img_in1.shape[0],:img_in1.shape[0]]

   downsampleSet1 = [img_in1]
   im1Backup =  img_in1.copy()
   for i in xrange(NO_OF_LEVELS+1):
       img_in1 = cv2.pyrDown(img_in1)
       downsampleSet1.append(img_in1)

   downsampleSet2 = [img_in2]
   im2Backup =  img_in2.copy()
   for i in xrange(NO_OF_LEVELS+1):
       img_in2 = cv2.pyrDown(img_in2)
       downsampleSet2.append(img_in2)

    #laplacian Level 1 = Gaussian L1 - pyrUp(Gaussian L2)
   laplacianSet1 = [downsampleSet1[NO_OF_LEVELS]]
   for i in xrange(NO_OF_LEVELS,0,-1):
       ithGaussian = downsampleSet1[i]
       pyredUpithGaussian = cv2.pyrUp(ithGaussian)
       i_1thGaussian = downsampleSet1[i-1]
       ithLaplacian = cv2.subtract(i_1thGaussian,pyredUpithGaussian)
       laplacianSet1.append(ithLaplacian)



   laplacianSet2 = [downsampleSet2[NO_OF_LEVELS]]
   for i in xrange(NO_OF_LEVELS,0,-1):
       ithGaussian = downsampleSet2[i]
       pyredUpithGaussian = cv2.pyrUp(ithGaussian)
       i_1thGaussian = downsampleSet2[i-1]
       ithLaplacian = cv2.subtract(i_1thGaussian,pyredUpithGaussian)
       laplacianSet2.append(ithLaplacian)

   stackedLaplacians = []
   for i in xrange(NO_OF_LEVELS+1):
       ithLap1 = laplacianSet1[i]
       ithLap2 = laplacianSet2[i]
       a,b,c = ithLap1.shape

       stackedLap = np.hstack((ithLap1[ : , :b/2 , : ], ithLap2[ : , b/2: , : ] ))
       stackedLaplacians.append(stackedLap)

   t = stackedLaplacians[0]
   #print  len(stackedLaplacians)
   for i in xrange(1, NO_OF_LEVELS+1):
       t = cv2.pyrUp(t)
       t = cv2.add(t, stackedLaplacians[i])
   img_out = t


   ###
   return True, img_out

def Question3():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);

   # Laplacian pyramid blending
   succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)

   # Write out the result
   output_name = sys.argv[4] + "output3.png"
   cv2.imwrite(output_name, output_image)

   return True

if __name__ == '__main__':
   question_number = -1

   # Validate the input arguments
   if (len(sys.argv) < 4):
      help_message()
      sys.exit()
   else:
      question_number = int(sys.argv[1])

      if (question_number == 1 and not(len(sys.argv) == 4)):
         help_message()
	 sys.exit()
      if (question_number == 2 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number == 3 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
	 print("Input parameters out of bound ...")
         sys.exit()

   function_launch = {
   1 : Question1,
   2 : Question2,
   3 : Question3,
   }

   # Call the function
   function_launch[question_number]()
