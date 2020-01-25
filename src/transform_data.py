import numpy as np
import cv2
from skimage import transform


def resize(img, size, order=1):
    """simple interpolation is better (more easily predict output)?"""
    return transform.resize(img, size, order=order)


def make_3channel(img):
    """Ensures input image is a 3 channel image for training/predicting"""
    img = np.reshape(img, (img.shape[0], img.shape[1], 1))
    img = np.concatenate((img, img, img), axis=2)

    return img


def reflect_pad(img, remove_coord):
    """
    Replaces region in img defined by remove_coord with a
    'reflection' of the region above and below remove_coord
    """
    remove_x1, remove_y1, remove_x2, remove_y2 = remove_coord
    remove_height = remove_y2 - remove_y1
    br_ylim = remove_y2 + remove_height // 2
    tr_ylim = remove_y1 - (remove_height - remove_height // 2)
    if br_ylim >= img.shape[0]:
        br_ylim = img.shape[0]
        tr_ylim = remove_y1 - (remove_height - (br_ylim - remove_y2))
    elif tr_ylim <= 0:
        tr_ylim = 0
        br_ylim = remove_y2 - (remove_height - remove_y1)
    bot_reflect = img[remove_y2:br_ylim, remove_x1:remove_x2, :]
    bot_reflect = np.flipud(bot_reflect)
    top_reflect = img[tr_ylim: remove_y1, remove_x1:remove_x2, :]
    top_reflect = np.flipud(top_reflect)
    reflect_pad = np.concatenate((top_reflect, bot_reflect), axis=0)
    imgcopy = img.copy()
    imgcopy[remove_y1:remove_y2, remove_x1:remove_x2] = reflect_pad
    return imgcopy


def replace_constant(img, remove_coord, constant):
    remove_x1, remove_y1, remove_x2, remove_y2 = remove_coord
    imgcopy = img.copy()
    imgcopy[remove_y1:remove_y2, remove_x1:remove_x2] = constant
    return imgcopy


def replace_neighbor(img, remove_coord):
    remove_x1, remove_y1, remove_x2, remove_y2 = remove_coord
    remove_height = remove_y2 - remove_y1
    imgcopy = img.copy()
    bot_neighbor = remove_y2 + 1
    top_neighbor = remove_y1 - 1
    if bot_neighbor == img.shape[0]:
        for i in range(remove_y1, remove_y2):
            imgcopy[i, remove_x1:remove_x2, :] = imgcopy[top_neighbor,
                                                         remove_x1:remove_x2,
                                                         :]
    elif top_neighbor == 0:
        for i in range(remove_y1, remove_y2):
            imgcopy[i, remove_x1:remove_x2, :] = imgcopy[bot_neighbor,
                                                         remove_x1:remove_x2,
                                                         :]
    else:
        ylim = remove_y2 - (remove_height // 2)
        for i in range(remove_y1, remove_y2):
            if i <= ylim:
                imgcopy[i, remove_x1:remove_x2, :] = imgcopy[top_neighbor,
                                                             remove_x1:remove_x2,
                                                             :]
            else:
                imgcopy[i, remove_x1:remove_x2, :] = imgcopy[bot_neighbor,
                                                             remove_x1:remove_x2,
                                                             :]
    return imgcopy


def find_remove_box(img, roi, threshold=250):
    x1, y1, x2, y2 = roi
    img = img[y1:y2, x1:x2, :]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # Find verticle and horizontal lines in image
    vert_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    hori_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_v = cv2.erode(thresh, vert_kern, iterations=3)
    vert_lines_img = cv2.dilate(img_v, vert_kern, iterations=5)
    img_h = cv2.erode(thresh, hori_kern, iterations=3)
    hori_lines_img = cv2.dilate(img_h, hori_kern, iterations=5)
    # Find remove box
    img_final = cv2.addWeighted(vert_lines_img, 0.5, hori_lines_img, 0.5, 0.0)
    img_final = cv2.erode(~img_final, kernel, iterations=2)
    ret, thresh2 = cv2.threshold(img_final,
                                 128,
                                 255,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if threshold >= 128:
        remove_box = np.argwhere(thresh2 == 0)
    else:
        remove_box = np.argwhere(thresh2 == 255)
    remove_box = (remove_box[0, 1] + x1,
                  remove_box[0, 0] + y1,
                  remove_box[-1, 1] + x1,
                  remove_box[-1, 0] + y1)
    return remove_box


def replace_whitewm_constant(img, roi, constant, threshold=225):
    x1, y1, x2, y2 = roi
    imgcopy = img.copy()
    img_roi = imgcopy[y1:y2, x1:x2, :]
    img_roi[img_roi >= threshold] = constant
    imgcopy[y1:y2, x1:x2, :] = img_roi
    return imgcopy


def replace_whitewm_avg(img, roi, threshold=225, method='mean'):
    x1, y1, x2, y2 = roi
    imgcopy = img.copy()
    img_roi = imgcopy[y1:y2, x1:x2, :]

    notwm = img_roi[img_roi < threshold]

    if method == 'mean':
        avg = np.mean(notwm)
    elif method == 'median':
        avg = np.median(notwm)
    else:
        raise ValueError('method must be "mean" or "median"')

    img_roi[img_roi >= threshold] = avg
    imgcopy[y1:y2, x1:x2, :] = img_roi
    return imgcopy


def whitewm_moving_avg(img, roi,
                       window_size=(5, 5), threshold=225, method='mean'):
    x1, y1, x2, y2 = roi
    imgcopy = img.copy()
    img_roi = imgcopy[y1:y2, x1:x2, :]

    for i in range(0, img_roi.shape[0], window_size[1]):
        for j in range(0, img_roi.shape[1], window_size[0]):
            window = img_roi[i:i+window_size[1], j:j+window_size[0], :]
            notwm = window[window < threshold]

            if method == 'mean':
                avg = np.mean(notwm)
            elif method == 'median':
                avg = np.median(notwm)
            else:
                raise ValueError('method must be "mean" or "median"')

            window[window >= threshold] = avg
            img_roi[i:i+window_size[1], j:j+window_size[0], :] = window

    imgcopy[y1:y2, x1:x2, :] = img_roi
    return imgcopy
