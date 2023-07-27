import cv2 as cv
import datetime

capture = cv.VideoCapture(0)
ret, img = capture.read()
capture.release()
cv.imshow("Images",img)
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
img_file = "Images/image"+timestamp+".jpg"
cv.imwrite(img_file, img)

cv.waitKey(0)
cv.destroyAllWindows()