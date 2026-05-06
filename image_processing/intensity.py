# 負責基本強度轉換

import cv2
import numpy as np

class IntensityMixin:
    
    # ==========================================
    # 🎨 輔助工具：繪製轉換函數曲線
    # ==========================================
    def _draw_transfer_curve(self, lookup_table, title="轉換函數曲線 (Transfer Function)"):
        """用 OpenCV 畫出 256x256 的轉換曲線圖"""
        # 建立深色背景畫布
        curve_img = np.full((256, 256, 3), 15, dtype=np.uint8)
        
        # 畫網格線
        for i in range(0, 256, 64):
            cv2.line(curve_img, (i, 0), (i, 255), (50, 50, 50), 1)
            cv2.line(curve_img, (0, i), (255, i), (50, 50, 50), 1)
            
        # 畫對角線 (基準線)
        cv2.line(curve_img, (0, 255), (255, 0), (100, 100, 100), 1, cv2.LINE_AA)

        # 畫轉換曲線
        for i in range(255):
            pt1 = (i, 255 - int(lookup_table[i]))
            pt2 = (i + 1, 255 - int(lookup_table[i + 1]))
            cv2.line(curve_img, pt1, pt2, (255, 0, 85), 2, cv2.LINE_AA)
            
        self.steps.append((title, curve_img))

    # ==========================================
    # ⚙️ 演算法實作 (加入步驟紀錄)
    # ==========================================
    def binarize(self, threshold=127, thresh_type='binary', use_otsu=False):
        gray = self._get_gray()
        self.steps.append(("Step 1: 轉換為灰階 (Grayscale)", gray))
        
        type_mapping = {
            'binary': cv2.THRESH_BINARY,
            'binary_inv': cv2.THRESH_BINARY_INV,
            'trunc': cv2.THRESH_TRUNC,
            'tozero': cv2.THRESH_TOZERO,
            'tozero_inv': cv2.THRESH_TOZERO_INV
        }
        
        # 1. 先保留基礎的二值化模式 (不含 Otsu)
        base_cv_type = type_mapping.get(thresh_type, cv2.THRESH_BINARY)
        cv_type = base_cv_type
        
        # 2. 如果啟用 Otsu，才在實際處理影像的 cv_type 加上標籤
        if use_otsu:
            cv_type += cv2.THRESH_OTSU
            
        # 3. 對影像進行二值化，取得實際使用的 actual_thresh
        actual_thresh, result = cv2.threshold(gray, threshold, 255, cv_type)
        
        # 4. 畫出二值化的階梯曲線 (⚠️ 這裡改用 base_cv_type，不要帶 Otsu 標籤)
        x = np.arange(256, dtype=np.uint8)
        _, lut = cv2.threshold(x, actual_thresh, 255, base_cv_type) 
        
        self._draw_transfer_curve(lut, f"Step 2: 二值化轉換曲線 (Thresh={int(actual_thresh)})")
        
        return result

    def linear_transform(self, alpha=1.0, beta=0):
        # 畫出線性轉換曲線
        x = np.arange(256, dtype=np.float32)
        lut = np.clip(alpha * x + beta, 0, 255).astype(np.uint8)
        self._draw_transfer_curve(lut, f"線性轉換曲線 (Alpha={alpha}, Beta={beta})")
        
        return cv2.convertScaleAbs(self.img, alpha=alpha, beta=beta)

    def negative_transform(self):
        # 畫出負轉換曲線
        x = np.arange(256, dtype=np.uint8)
        lut = 255 - x
        self._draw_transfer_curve(lut, "負轉換曲線 (Negative Mapping)")
        
        return 255 - self.img

    def log_transform(self, c=1.0):
        # 畫出對數轉換曲線
        x = np.arange(256, dtype=np.float32)
        lut = c * np.log(1 + x)
        lut = np.uint8(cv2.normalize(lut, None, 0, 255, cv2.NORM_MINMAX))
        self._draw_transfer_curve(lut, f"對數轉換曲線 (Log Curve, c={c})")
        
        result = c * np.log(1 + np.float32(self.img))
        return np.uint8(cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX))

    def power_law_transform(self, gamma=1.0, c=1.0):
        # 畫出 Gamma 轉換曲線
        x = np.arange(256, dtype=np.float32)
        lut = c * np.power(x / 255.0, gamma)
        lut = np.uint8(np.clip(lut * 255.0, 0, 255))
        self._draw_transfer_curve(lut, f"指數轉換曲線 (Gamma={gamma})")
        
        result = c * np.power(np.float32(self.img) / 255.0, gamma)
        return np.uint8(np.clip(result * 255.0, 0, 255))

    def histogram_equalization(self):
        # 擷取等化前的直方圖來計算 CDF
        gray = self._get_gray()
        hist, _ = np.histogram(gray.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf.max()
        self._draw_transfer_curve(cdf_normalized, "累積分布函數 (CDF Mapping)")

        if self.is_color:
            hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return cv2.equalizeHist(self.img)

    def clahe_equalization(self, clip_limit=2.0, tile_grid_size=8):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        if self.is_color:
            hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return clahe.apply(self.img)
