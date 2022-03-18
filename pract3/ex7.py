# file practical_3_exercise_4.py
import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np
img = image.imread("chat.jpg")
print(img.shape) # check the size of the image
U, s, Vs = np.linalg.svd(np.moveaxis(img,-1,0), full_matrices=False)
# reconstruct the image
k = 20 # rank of new image. 8 is quite low!
lowrank = np.moveaxis( (U[...,:k] * s[...,None,:k]) @ Vs[...,:k,:], 0, -1)
# img has range from 0 to 255 and ce be stored in 8 bit (uint8) integers
# lowrank has floating point values, but are not constrained in [0, 255]
lowrank = np.array(lowrank, "int16") # cast to signed integer 16
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img) # create a subplot for the original
ax1.set_title("Original")
ax2.imshow(lowrank) # create a subplot for the low rank approx
ax2.set_title("Low-rank")
plt.show()
