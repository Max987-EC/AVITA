# 負責基礎設定與直方圖/頻譜圖

import cv2
import numpy as np
import base64

class BaseProcessor:
    def __init__(self, img_array):
        self.img = img_array
        self.is_color = len(img_array.shape) == 3
        self.steps = [] 

    def add_step(self, name, img):
        self.steps.append((name, img.copy()))

    def _get_gray(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) if self.is_color else self.img

    @staticmethod
    def generate_histogram_base64(img):
        h, w = 300, 400
        pad_bottom = 30 
        hist_img = np.zeros((h + pad_bottom, w, 3), dtype=np.uint8)
        hist_img[:] = (25, 15, 10) 
        cv2.line(hist_img, (0, h), (w, h), (100, 100, 100), 1)

        is_gray = False
        if len(img.shape) == 2 or img.shape[2] == 1:
            is_gray = True
        elif len(img.shape) == 3:
            b, g, r = cv2.split(img)
            color_mask = cv2.bitwise_or(cv2.absdiff(b, g), cv2.absdiff(g, r))
            if cv2.countNonZero(color_mask) / (img.shape[0] * img.shape[1]) < 0.5:
                is_gray = True
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if is_gray:
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, h, cv2.NORM_MINMAX)
            for x in range(1, 256):
                cv2.line(hist_img, (int((x - 1) * w / 256), h - int(hist[x - 1][0])),
                         (int(x * w / 256), h - int(hist[x][0])), (150, 150, 150), 2)
        else:
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] 
            for i, col in enumerate(colors):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                cv2.normalize(hist, hist, 0, h, cv2.NORM_MINMAX)
                for x in range(1, 256):
                    cv2.line(hist_img, (int((x - 1) * w / 256), h - int(hist[x - 1][0])),
                             (int(x * w / 256), h - int(hist[x][0])), col, 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        for val in range(0, 257, 16):
            if val == 256: val = 255
            x_pos = int(val * (w - 1) / 255)
            if val % 32 == 0 or val == 255:
                cv2.line(hist_img, (x_pos, h), (x_pos, h + 8), (180, 180, 180), 1)
                text = str(val)
                (tw, _), _ = cv2.getTextSize(text, font, 0.35, 1)
                tx = max(2, min(x_pos - (tw // 2), w - tw - 2))
                cv2.putText(hist_img, text, (tx, h + 22), font, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
            else:
                cv2.line(hist_img, (x_pos, h), (x_pos, h + 4), (100, 100, 100), 1)

        _, buffer = cv2.imencode('.png', hist_img)
        return base64.b64encode(buffer).decode('utf-8')

    @staticmethod
    def generate_spectrum_base64(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        fshift = np.fft.fftshift(np.fft.fft2(gray))
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        spectrum_img = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, buffer = cv2.imencode('.jpg', spectrum_img)
        return base64.b64encode(buffer).decode('utf-8')
