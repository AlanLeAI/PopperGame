import numpy as np
import cv2

def threshold_frame(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
    img = cv2.GaussianBlur(img, (5,5), 1)
    _, img = cv2.threshold(img,200,255,cv2.THRESH_BINARY_INV)
    return img

def get_structure_elements(file_path):
    se = cv2.imread(file_path)
    se_resized = cv2.resize(se, (100, 120))
    _, se_thresh = cv2.threshold(se_resized,50,255,cv2.THRESH_BINARY_INV)
    mask = cv2.cvtColor(se_thresh, cv2.COLOR_BGR2GRAY)
    return mask

def find_contours(img):
    h,w= img.shape
    img_contour = np.zeros((h,w,3), np.int8)
    contours, hierachy = cv2.findContours(img ,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    cv2.drawContours(img,contours, -1, (255,0,255),2)
    bounding_boxes = []
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area  > 1000:
            cv2.drawContours(img_contour, contours, i , (255,0,255), 2)
            x,y,w,h = cv2.boundingRect(cnt)
            bounding_boxes.append([(x,y), (x+w,y+h)])
            cv2.rectangle(img_contour, (x,y), (x+w,y+h), (0,255,0),2)
    return img_contour, bounding_boxes