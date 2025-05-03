import numpy as np
import math
import cv2


def threshold_frame(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 1)
    _, img = cv2.threshold(img, 165, 255, cv2.THRESH_BINARY_INV)
    return img


def get_structure_elements(file_path, size=None):
    se = cv2.imread(file_path)
    if size:
        se_resized = cv2.resize(se, size)
        lower_black_bgr = (0, 0, 0)
        upper_black_bgr = (100, 100, 100)
        mask = cv2.inRange(se_resized, lower_black_bgr, upper_black_bgr, cv2.THRESH_BINARY_INV)
    else:
        lower_black_bgr = (0, 0, 0)
        upper_black_bgr = (100, 100, 100)
        mask = cv2.inRange(se, lower_black_bgr, upper_black_bgr, cv2.THRESH_BINARY_INV)
    return mask


def find_contours(img):
    h, w = img.shape
    img_contour = np.zeros((h, w, 3), np.int8)
    contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 5000:
            cv2.drawContours(img_contour, contours, i, (255, 0, 255), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 5000:
                bounding_boxes.append([(x, y), (x + w, y + h)])
                cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
        return se


def crop_se_hori(se):
    white_cols = np.where(se.any(axis=0))[0]
    if len(white_cols) > 0:
        first_white_col = white_cols[0]
        last_white_col = white_cols[-1]
        return se[:, first_white_col:last_white_col + 1]
    else:
        return se


def detect_energy_balloon(balloon, box_size):
    balloon = cv2.morphologyEx(balloon, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    cropped_mask = crop_blank_spaces(balloon)
    mask = cv2.dilate(cropped_mask, np.ones((3, 3), dtype=np.uint8), iterations=4)

    se_energy = get_structure_elements("images/energy1.png")
    cropped_se = crop_blank_spaces(se_energy)
    erode_se = cv2.erode(cropped_se, np.ones((2, 2), dtype=np.uint8), iterations=3)
    erode_se = cv2.resize(erode_se, (mask.shape[1], mask.shape[0]))
    match = cv2.erode(mask, erode_se)
    if sum(sum(match)) > 5:
        return True
    else:
        return False


def detect_bomb_balloon(balloon, box_size):
    mask = cv2.morphologyEx(balloon, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    cropped_mask = crop_blank_spaces(mask)
    mask = cv2.dilate(cropped_mask, np.ones((3, 3), dtype=np.uint8), iterations=2)

    se_bomb = get_structure_elements("images/bomb_1.png", box_size)
    cropped_se = crop_blank_spaces(se_bomb)
    erode_se = cv2.erode(cropped_se, np.ones((3, 3), dtype=np.uint8))
    match = cv2.erode(mask, erode_se)
    if sum(sum(match)) > 5:
        return True
    else:
        return False


def detect_number_balloon(balloon, box_size):
    balloon = cv2.morphologyEx(balloon, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    cropped_mask = crop_blank_spaces(balloon)
    mask = cv2.dilate(cropped_mask, np.ones((3, 3), dtype=np.uint8), iterations=4)

    se_number = get_structure_elements("images/regular4.png")
    cropped_se = crop_blank_spaces(se_number)
    erode_se = cv2.erode(cropped_se, np.ones((2, 2), dtype=np.uint8), iterations=3)
    erode_se = cv2.resize(erode_se, (mask.shape[1], mask.shape[0]))
    match = cv2.erode(mask, erode_se)
    if sum(sum(match)) > 5:
        return True
    else:
        return False


def detect_ballon(frame, bounding_boxes, balloons, size=None):
    result = {}
    for box in bounding_boxes:
        if len(bounding_boxes) > 0:
            x1, y1 = box[0]
            x2, y2 = box[1]

            box_width = x2 - x1
            box_height = y2 - y1

            if box_width > 300 or box_height > 300:
                continue

            extract_balloon = frame[y1:y2, x1:x2, :]
            lower_black_bgr = (0, 0, 0)
            upper_black_bgr = (140, 140, 140)
            mask = cv2.inRange(extract_balloon, lower_black_bgr, upper_black_bgr, cv2.THRESH_BINARY_INV)
            box_size = (box_width, box_height)
            cv2.imshow("extract_balloon_mask", mask)
            if detect_bomb_balloon(mask, box_size):
                label = "bomb"
            elif detect_energy_balloon(mask, box_size):
                label = "energy"
            elif detect_number_balloon(mask, box_size):
                label = "number"
            else:
                label = "regular"

            closest_balloon = None
            min_distance = float('inf')
            for balloon in balloons:
                distance = math.sqrt((x1 - balloon.x) ** 2 + (y1 - balloon.y) ** 2)
                if distance < min_distance and label in balloon.type:
                    min_distance = distance
                    closest_balloon = balloon

            if closest_balloon:
                result[(x1, y1, x2, y2)] = closest_balloon
    return result


def detect_yellow_obj(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, (10, 70, 70), (30, 255, 255))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    yellow_mask = cv2.dilate(yellow_mask, np.ones((30, 30), np.uint8), iterations=1)
    cv2.imshow("yellow_mask", yellow_mask)

    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        combined_contour = np.vstack(contours)

        x, y, w, h = cv2.boundingRect(combined_contour)

        if w > 50 and w < 150 and h > 50 and h < 150:

            print(f"yellow_obj: {x}, {y}, {w}, {h}")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            center_x = x + w // 2
            center_y = y + h // 2

            temp = frame.copy()
            temp[yellow_mask != 255] = (0, 0, 0)
            cv2.imshow("yellow_obj", temp)

            return center_x, center_y
    return None, None


def detect_collision(center, mapping):
    if center is None:
        return None

    cx, cy = center

    try:
        for (x1, y1, x2, y2), balloon in mapping.items():
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return (x1, y1, x2, y2), balloon
    except Exception as ex:
        return None

    return None
