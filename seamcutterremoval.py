import seamcut;
import cv2;
import numpy as np;


# Read the npy file
mask = np.load("mask.npy")

# Read the image
image = cv2.imread("surfer.jpg")

# initialize the seam cutter
sc = seamcut.SeamCarver(image)

# modify the energy matrix so the masked parts will be removed as part of the seam carving process
# this can be done by setting the energy of the masked parts to 0
sc.energy_matrix()[mask == 1] = 0


# wait we have to modify the seam finding code to accept energy matrix as an argument instead of calculating it itself

# Perform seam carving
