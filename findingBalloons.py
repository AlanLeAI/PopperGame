import numpy as np
import cv2


def find_balloons(img):
    # img = crop_image(img,0.02)
    img = preprocess_img(img)
    imgcontour = find_contours(img)
    return imgcontour

def crop_image(img, crop_val):
    h,w,c = img.shape
    img = img[ 0:w, int(crop_val*h):h,]
    return img

def preprocess_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
    img = cv2.GaussianBlur(img, (5,5), 1)
    img = cv2.Canny(img,50,50)
    kernel = np.ones((5,5), np.uint8)
    img = cv2.dilate(img,kernel)
    return img

def find_contours(img):
    h,w= img.shape
    img_contour = np.zeros((w,h,3), np.int8)
    contours, hierachy = cv2.findContours(img ,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    cv2.drawContours(img,contours, -1, (255,0,255),2)
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        # print(area)
        if area  > 20000 and area < 250000:
            cv2.drawContours(img_contour, contours, i , (255,0,255), 2)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(img_contour, (x,y), (x+w,y+h), (0,255,0),2)
    return img_contour