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

def crop_blank_spaces(se):
    crop_vert = crop_se_vert(se)
    return crop_se_hori(crop_vert)

def crop_se_vert(se):
    white_rows = np.where(se.any(axis=1))[0]
    if len(white_rows) > 0:
        first_white_row = white_rows[0]
        last_white_row = white_rows[-1]
        return se[first_white_row:last_white_row + 1, :]
    else:
        print("No white pixels found in the mask")
        return se

def crop_se_hori(se):
    white_cols = np.where(se.any(axis=0))[0]
    if len(white_cols) > 0:
        first_white_col = white_cols[0]
        last_white_col = white_cols[-1]
        return se[:, first_white_col:last_white_col + 1]
    else:
        print("No white pixels found in the mask")
        return se

def detect_energy_balloon(balloon):
    se_energy = get_structure_elements("images/energy1.png")
    cropped_mask = crop_blank_spaces(balloon)
    mask = cv2.dilate(cropped_mask, np.ones((2, 2), dtype=np.uint8))
    # cv2.imshow("cropped_mask", mask)
    cropped_se = crop_blank_spaces(se_energy)
    erode_se = cv2.erode(cropped_se, np.ones((2, 2), dtype=np.uint8))
    match = cv2.erode(mask, erode_se)
    cv2.imshow("match", match)
    print(sum(sum(match)))
    if sum(sum(match)) > 100:
        return True
    else:
        return False

def detect_ballon(frame, bounding_boxes):
    result = {}
    for box in bounding_boxes:
        if len(bounding_boxes) > 0:
            x1, y1 = box[0]
            x2, y2 = box[1]
            extract_balloon = frame[y1:y2, x1:x2, :]
            lower_black_bgr = (0, 0, 0)
            upper_black_bgr = (100, 100, 100)
            mask = cv2.inRange(extract_balloon, lower_black_bgr, upper_black_bgr, cv2.THRESH_BINARY_INV)
            cv2.imshow("mask", mask)
            if detect_energy_balloon(mask):
                result[(x1,y1,x2,y2)] = "energy"

    print(result)
    return result