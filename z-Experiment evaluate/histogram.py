import cv2
import numpy as np
import matplotlib.pyplot as plt

## paint histogram image
image_path = 'data/cipherimages/0_1.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# calculate histogram picture
hist_r, bins_r = np.histogram(image_rgb[:, :, 0].ravel(), bins=256, range=(0, 256))
hist_g, bins_g = np.histogram(image_rgb[:, :, 1].ravel(), bins=256, range=(0, 256))
hist_b, bins_b = np.histogram(image_rgb[:, :, 2].ravel(), bins=256, range=(0, 256))

plt.figure(figsize=(8, 5))

plt.plot(bins_r[:-1], hist_r, color='r', label='Red')
plt.plot(bins_g[:-1], hist_g, color='g', label='Green')
plt.plot(bins_b[:-1], hist_b, color='b', label='Blue')

plt.xlim(0, 255)
plt.ylim(0, max(hist_r.max()+100, hist_g.max(), hist_b.max()))
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend(loc='upper right')

ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_position(('outward', 0))
ax.spines['top'].set_position(('outward', 0))
ax.spines['right'].set_color('k')
ax.spines['top'].set_color('k')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.show()
