import numpy as np
import cv2

img = cv2.imread('grass2.jpg')
vect = img.reshape((-1,3))
vect = np.float32(vect) # convert to float for k-means calculation
k=8
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, centroids = cv2.kmeans(vect, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centroids = np.uint8(centroids) # convert back to image values
outpic = centroids[label.flatten()]
outpic2 = outpic.reshape((img.shape))
cv2.imwrite('newgrass3.jpg', outpic2)

