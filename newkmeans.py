import numpy as np
import cv2

def kmeans_pic(inpicfile, outpicfile, k):
	img = cv2.imread(inpicfile)
	vect = img.reshape((-1,3))
	vect = np.float32(vect) # convert to float for k-means calculation

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	_, label, centroids = cv2.kmeans(vect, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

	centroids = np.uint8(centroids) # convert back to image values
	outpic = centroids[label.flatten()]
	outpic2 = outpic.reshape((img.shape))
	cv2.imwrite(outpicfile, outpic2)

kmeans_pic('grass2.jpg', 'newgrass3.jpg', 3)
#kmeans_pic('grass2.jpg', 'outgrass5.jpg', 5)
#kmeans_pic('grass2.jpg', 'outgrass8.jpg', 8)
