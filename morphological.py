import cv2
import numpy as np
### For good-balloon
# img_original = cv2.imread('./test/191.conv.jpg')
# img_pred = cv2.imread('./test/191.conv_pred.jpg',0)
# ret, thresh = cv2.threshold(img_pred, 127, 255, 0)
# image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
# for c in contours:
#     # get the bounding rect
#     x, y, w, h = cv2.boundingRect(c)
#     # draw a white rectangle to visualize the bounding rect
#     cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 255, 0), 2)

# cv2.drawContours(img_original, contours, -1, (255, 0, 0), 2)

MIN_AREA = 5000

def getBoundingBox(pts):
	min_x=min_y=5000
	max_x=max_y= 0 
	for point in pts:
		coord = point[0]
		coord_x, coord_y = coord
		min_x = min_x if min_x < coord_x else coord_x
		min_y = min_y if min_y <= coord_y else coord_y
		max_x = max_x if max_x > coord_x else coord_x
		max_y = max_y if max_y >= coord_y else coord_y
	width = max_x - min_x
	height = max_y - min_y
	return min_x, min_y, width, height

def getChildBoundingBox(bbox):
	x, y, width, height = bbox


### For implicit-balloon # 185
img_original = cv2.imread('./test/206.conv.jpg')
img_pred = cv2.imread('./test/206.conv_pred.jpg',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img_pred,kernel,iterations = 1)
img_pred = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
ret, thresh = cv2.threshold(img_pred, 127, 255, 0)
image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
for contour in contours:
	cnt = contour
	area = cv2.contourArea(cnt)
	if area >= MIN_AREA:
		hull = cv2.convexHull(cnt)
		pts = np.array(hull, np.int32)
		pts = pts.reshape((-1,1,2))

		cv2.polylines(img_original, [pts], True, (0,255,0))
		bbox = getBoundingBox(pts)

		cv2.rectangle(img_original, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)

cv2.imshow("orginal",img_original)
cv2.imshow("img_segmentation",img_pred)
cv2.waitKey(0)
cv2.destroyAllWindows()
