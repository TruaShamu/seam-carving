import numpy as np
import matplotlib.pyplot as plt

# load the mask from npy file
mask = np.load("surfer.npy")

# display the mask
plt.imshow(mask, cmap='gray')
plt.show()

print(np.unique(mask))

# convert the mask to boolean
mask = mask.astype(bool)

# print all the values in the mask
print(mask)