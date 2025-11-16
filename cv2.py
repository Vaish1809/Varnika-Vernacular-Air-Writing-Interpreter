# Minimal stub for cv2 so mediapipe can import drawing_utils on Streamlit
# without loading the real OpenCV binary (which needs libGL).

import numpy as np

# Common constants mediapipe.drawing_utils might reference
COLOR_BGR2RGB = 4
LINE_AA = 16
FILLED = -1

def cvtColor(img, code):
    # Only support BGR -> RGB; everything else is returned unchanged
    if code == COLOR_BGR2RGB:
        return img[..., ::-1]
    return img

# Drawing functions are no-ops because we don't use mediapipe's drawing
def circle(*args, **kwargs):
    pass

def line(*args, **kwargs):
    pass

def rectangle(*args, **kwargs):
    pass

# GUI-related functions as no-ops
def imshow(*args, **kwargs):
    pass

def waitKey(*args, **kwargs):
    return -1
