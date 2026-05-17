# 基礎影像處理器核心 (Core Base Processor)
# 此模組定義了所有處理器的基底類別，包含基礎設定、直方圖產生與頻譜分析。

import cv2
import numpy as np
import base64

class BaseProcessor:
    """
    影像處理基底類別 (Base Class for Image Processing):
    管理原始影像、顏色狀態、處理步驟紀錄，並提供統計分析工具。
    """
    def __init__(self, img_array):
        self.img = img_array
        # 判斷影像是否為彩色 (形狀是否包含三個通道)
        self.is_color = len(img_array.shape) == 3
        # 儲存處理過程中的中間步驟 (名稱, 影像數據)
        self.steps = [] 

    def add_step(self, name, img):
        """手動新增處理步驟"""
        self.steps.append((name, img.copy()))

    def _get_gray(self):
        """內部工具：獲取目前的灰階版本影像"""
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) if self.is_color else self.img

    @staticmethod
    def generate_histogram_base64(img):
        """
        產生直方圖影像並轉換為 Base64 字串 (Histogram Generation):
        支援彩色 (RGB) 與灰階影像。
        """
        h, w = 300, 400
        pad_bottom = 30 
        hist_img = np.zeros((h + pad_bottom, w, 3), dtype=np.uint8)
        hist_img[:] = (25, 15, 10) # 設定深色背景
        
        # 繪製 X 軸基準線
        cv2.line(hist_img, (0, h), (w, h), (100, 100, 100), 1)

        # 判斷輸入影像是否為灰階 (或數值上接近灰階)
        is_gray = False
        if len(img.shape) == 2 or img.shape[2] == 1:
            is_gray = True
        elif len(img.shape) == 3:
            b, g, r = cv2.split(img)
            # 透過計算通道間的差異來判斷是否為實質灰階
            color_mask = cv2.bitwise_or(cv2.absdiff(b, g), cv2.absdiff(g, r))
            if cv2.countNonZero(color_mask) / (img.shape[0] * img.shape[1]) < 0.5:
                is_gray = True
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if is_gray:
            # 1. 處理灰階直方圖
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, h, cv2.NORM_MINMAX)
            for x in range(1, 256):
                cv2.line(hist_img, (int((x - 1) * w / 256), h - int(hist[x - 1][0])),
                         (int(x * w / 256), h - int(hist[x][0])), (150, 150, 150), 2)
        else:
            # 2. 處理彩色三通道直方圖 (BGR)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] 
            for i, col in enumerate(colors):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                cv2.normalize(hist, hist, 0, h, cv2.NORM_MINMAX)
                for x in range(1, 256):
                    cv2.line(hist_img, (int((x - 1) * w / 256), h - int(hist[x - 1][0])),
                             (int(x * w / 256), h - int(hist[x][0])), col, 2)
        
        # 繪製刻度值與標記
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

        # 編碼影像為 PNG 格式並轉為 Base64
        _, buffer = cv2.imencode('.png', hist_img)
        return base64.b64encode(buffer).decode('utf-8')

    @staticmethod
    def generate_spectrum_base64(img):
        """
        產生頻譜圖影像 (Spectrum Analysis):
        執行二維快速傅立葉轉換 (FFT) 並將結果視覺化為 Base64 字串。
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        # 1. FFT 轉換
        f = np.fft.fft2(gray)
        # 2. 將零頻率成分移動到頻譜中心
        fshift = np.fft.fftshift(f)
        # 3. 計算幅度譜並進行對數縮放 (讓細節更清楚)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # 4. 正規化至 0-255 以利顯示
        spectrum_img = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        _, buffer = cv2.imencode('.jpg', spectrum_img)
        return base64.b64encode(buffer).decode('utf-8')
