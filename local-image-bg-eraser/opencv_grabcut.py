import cv2
import numpy as np
from io import BytesIO

def remove_background_grabcut(image):
    if isinstance(image, str):
        img = cv2.imread(image)
    elif hasattr(image, 'read'):
        image_bytes = image.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        image.seek(0)
    else:
        img = np.array(image)

    if img is None:
        raise ValueError("Image not found or invalid format")

    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (50, 50, img.shape[1]-100, img.shape[0]-100)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    output = img * mask2[:, :, np.newaxis]

    b, g, r = cv2.split(output)
    alpha = (mask2 * 255).astype(np.uint8)
    rgba = cv2.merge((r, g, b, alpha))

    return rgba
