# 強度轉換與點處理模組 (Intensity Transformations Module)
# 包含對單個像素進行變換的各種點處理技術 (Point Processing)。

import cv2
import numpy as np

class IntensityMixin:
    """
    強度轉換混入類別 (Intensity Transformation Mixin):
    提供二值化、線性/非線性變換 (Log, Gamma) 與直方圖均衡化等功能。
    """
    
    # ==========================================
    # 🎨 輔助工具：繪製轉換函數曲線 (Transfer Function Plot)
    # ==========================================
    def _draw_transfer_curve(self, lookup_table, title="轉換函數曲線 (Transfer Function)"):
        """
        利用 OpenCV 繪製 256x256 的強度映射曲線圖:
        視覺化輸入灰階值 (X軸) 如何對應到輸出灰階值 (Y軸)。
        """
        # 建立深色背景畫布
        curve_img = np.full((256, 256, 3), 15, dtype=np.uint8)
        
        # 繪製參考網格線
        for i in range(0, 256, 64):
            cv2.line(curve_img, (i, 0), (i, 255), (50, 50, 50), 1)
            cv2.line(curve_img, (0, i), (255, i), (50, 50, 50), 1)
            
        # 繪製 45 度對角線作為基準 (恆等變換)
        cv2.line(curve_img, (0, 255), (255, 0), (100, 100, 100), 1, cv2.LINE_AA)

        # 根據查表 (LUT) 繪製實際的轉換曲線
        for i in range(255):
            pt1 = (i, 255 - int(lookup_table[i]))
            pt2 = (i + 1, 255 - int(lookup_table[i + 1]))
            cv2.line(curve_img, pt1, pt2, (255, 0, 85), 2, cv2.LINE_AA)
            
        self.steps.append((title, curve_img))

    # ==========================================
    # ⚙️ 門檻化與二值化 (Thresholding)
    # ==========================================
    def binarize(self, threshold=127, thresh_type='binary', use_otsu=False):
        """
        全域門檻化 (Global Thresholding):
        將影像轉為黑白二值。支援固定門檻與 Otsu 自動門檻法。
        """
        gray = self._get_gray()
        self.steps.append(("Step 1: 轉換為灰階 (Grayscale)", gray))
        
        type_mapping = {
            'binary': cv2.THRESH_BINARY,
            'binary_inv': cv2.THRESH_BINARY_INV,
            'trunc': cv2.THRESH_TRUNC,
            'tozero': cv2.THRESH_TOZERO,
            'tozero_inv': cv2.THRESH_TOZERO_INV
        }
        
        base_cv_type = type_mapping.get(thresh_type, cv2.THRESH_BINARY)
        cv_type = base_cv_type
        
        # 如果啟用 Otsu，則讓 OpenCV 自動計算最佳門檻
        if use_otsu:
            cv_type += cv2.THRESH_OTSU
            
        actual_thresh, result = cv2.threshold(gray, threshold, 255, cv_type)
        
        # 產生轉換曲線視覺化
        x = np.arange(256, dtype=np.uint8)
        _, lut = cv2.threshold(x, actual_thresh, 255, base_cv_type) 
        
        self._draw_transfer_curve(lut, f"Step 2: 二值化轉換曲線 (Thresh={int(actual_thresh)})")
        self.steps.append(("Step 3: 二值化結果", result))
        
        return result

    def adaptive_threshold(self, max_value=255, adaptive_method='gaussian', thresh_type='binary', block_size=11, C=2):
        """
        自適應門檻化 (Adaptive Thresholding):
        針對影像各個區域動態計算門檻，有效解決光照不均的問題。
        """
        gray = self._get_gray()
        self.steps.append(("Step 1: 轉換為灰階 (Grayscale)", gray))
        
        # 參數校正：block_size 必須是奇數
        if block_size % 2 == 0: block_size += 1
        if block_size < 3: block_size = 3
            
        cv_adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if adaptive_method == 'gaussian' else cv2.ADAPTIVE_THRESH_MEAN_C
        cv_thresh_type = cv2.THRESH_BINARY_INV if thresh_type == 'binary_inv' else cv2.THRESH_BINARY
        
        result = cv2.adaptiveThreshold(
            gray, max_value, cv_adaptive_method, cv_thresh_type, block_size, C
        )
        
        method_name = "高斯權重" if adaptive_method == 'gaussian' else "均值權重"
        self.steps.append((f"Step 2: 自適應二值化 ({method_name}, Block={block_size})", result))
        
        return result

    def linear_transform(self, alpha=1.0, beta=0):
        """
        線性轉換 (Linear Transformation):
        g(x,y) = alpha * f(x,y) + beta。用於調整對比度 (alpha) 與亮度 (beta)。
        """
        x = np.arange(256, dtype=np.float32)
        lut = np.clip(alpha * x + beta, 0, 255).astype(np.uint8)
        self._draw_transfer_curve(lut, f"Step 1: 線性轉換曲線 (Alpha={alpha}, Beta={beta})")
        
        result = cv2.convertScaleAbs(self.img, alpha=alpha, beta=beta)
        self.steps.append(("Step 2: 線性轉換結果", result))
        return result

    def negative_transform(self):
        """
        負片轉換 (Negative Transformation):
        s = L - 1 - r。反轉影像顏色。
        """
        x = np.arange(256, dtype=np.uint8)
        lut = 255 - x
        self._draw_transfer_curve(lut, "Step 1: 負轉換曲線 (Negative Mapping)")
        
        result = 255 - self.img
        self.steps.append(("Step 2: 負轉換結果", result))
        return result

    def log_transform(self, c=1.0):
        """
        對數轉換 (Log Transformation):
        s = c * log(1 + r)。用於壓縮高動態範圍影像，擴展暗部細節。
        """
        # 自動計算基準常數，將輸入 255 映射至輸出 255
        base_c = 255.0 / np.log(256.0)
        
        x = np.arange(256, dtype=np.float32)
        lut = c * base_c * np.log(1 + x)
        lut = np.uint8(np.clip(lut, 0, 255))
        self._draw_transfer_curve(lut, f"Step 1: 對數轉換曲線 (c={c})")
        
        result = c * base_c * np.log(1 + np.float32(self.img))
        result = np.uint8(np.clip(result, 0, 255))
        
        self.steps.append((f"Step 2: 對數轉換結果", result))
        return result

    def power_law_transform(self, gamma=1.0, c=1.0):
        """
        冪次/伽瑪轉換 (Power-Law / Gamma Transformation):
        s = c * r^gamma。用於校正顯示設備的非線性特性或調整對比度。
        """
        x = np.arange(256, dtype=np.float32)
        lut = c * np.power(x / 255.0, gamma)
        lut = np.uint8(np.clip(lut * 255.0, 0, 255))
        self._draw_transfer_curve(lut, f"Step 1: 指數轉換曲線 (Gamma={gamma})")
        
        result = c * np.power(np.float32(self.img) / 255.0, gamma)
        result = np.uint8(np.clip(result * 255.0, 0, 255))
        
        self.steps.append((f"Step 2: 指數轉換結果", result))
        return result

    # ==========================================
    # ⚙️ 直方圖處理 (Histogram Processing)
    # ==========================================
    def histogram_equalization(self):
        """
        直方圖均衡化 (Histogram Equalization):
        自動調整影像強度，使分佈更均勻，提升整體對比度。
        """
        gray = self._get_gray()
        # 計算累積分布函數 (CDF) 作為轉換函數
        hist, _ = np.histogram(gray.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf.max()
        self._draw_transfer_curve(cdf_normalized, "Step 1: 累積分布函數 (CDF Mapping)")

        if self.is_color:
            # 彩色影像先轉 HSV 空間，僅對亮度 (V) 通道處理以保持色調
            hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            result = cv2.equalizeHist(self.img)
            
        self.steps.append(("Step 2: 直方圖均衡化結果", result))
        return result

    def clahe_equalization(self, clip_limit=2.0, tile_grid_size=8):
        """
        限制對比度自適應直方圖均衡化 (CLAHE):
        局部處理影像的不同區域，並限制對比度增長，避免過度放大雜訊。
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        
        if self.is_color:
            hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            self.steps.append(("Step 1: 擷取亮度通道 (V Channel)", v_channel.copy()))
            
            processed_channel = clahe.apply(v_channel)
            hsv[:, :, 2] = processed_channel
            final_result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            gray = self.img.copy()
            self.steps.append(("Step 1: 原始灰階影像", gray.copy()))
            processed_channel = clahe.apply(gray)
            final_result = processed_channel

        self.steps.append((f"Step 2: 局部等化結果 (Clip={clip_limit})", processed_channel.copy()))
        
        # 視覺化整體 CDF 的改變
        hist, _ = np.histogram(processed_channel.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf.max()
        self._draw_transfer_curve(cdf_normalized, "Step 3: 處理後整體 CDF 狀態")
        
        self.steps.append(("Step 4: CLAHE 最終結果", final_result))
        return final_result

    def moving_average_threshold(self, n=20, k=0.5):
        """
        移動平均門檻化 (Moving Average Thresholding):
        模擬掃描線過程，根據先前像素的平均值來決定當前門檻。適用於掃描文件影像。
        """
        gray = self._get_gray()
        self.steps.append(("Step 1: 轉換為灰階 (Grayscale)", gray))
        
        # 建立 1D 水平移動平均濾波器
        kernel = np.ones((1, n), np.float32) / n
        # 設定 anchor 以便只對「左方 (過去)」像素進行加權
        moving_avg = cv2.filter2D(gray.astype(np.float32), -1, kernel, anchor=(n-1, 0))
        
        # 判斷邏輯：像素值 > k * 均值
        binary = np.where(gray > k * moving_avg, 255, 0).astype(np.uint8)
        
        self.steps.append((f"Step 2: 動態門檻化 (N={n}, k={k})", binary))
        return binary

    def edge_improved_threshold(self):
        """
        邊緣輔助全域門檻化 (Edge-Improved Thresholding):
        僅利用物體邊緣附近的像素來計算直方圖與 Otsu 門檻，能更精準地分離目標與背景。
        """
        gray = self._get_gray()
        self.steps.append(("Step 1: 原始灰階影像", gray))
        
        # 1. 取得梯度強度以標記邊緣區域
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.steps.append(("Step 2: 邊緣梯度強度圖", magnitude))
        
        # 2. 建立邊緣遮罩
        _, edge_mask = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
        self.steps.append(("Step 3: 邊緣遮罩 (僅保留邊界點)", edge_mask))
        
        # 3. 在遮罩範圍內手動執行 Otsu 演算法計算門檻
        hist = cv2.calcHist([gray], [0], edge_mask, [256], [0, 256])
        hist_norm = hist.ravel() / (hist.sum() + 1e-7)
        Q = hist_norm.cumsum()
        bins = np.arange(256)
        fn_min = np.inf
        thresh = 0
        
        for i in range(1, 256):
            p1, p2 = np.hsplit(hist_norm, [i])
            q1, q2 = Q[i], Q[255] - Q[i]
            if q1 == 0 or q2 == 0: continue
            b1, b2 = np.hsplit(bins, [i])
            m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
            v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2
            fn = v1 * q1 + v2 * q2 # 類內方差
            if fn < fn_min:
                fn_min, thresh = fn, i
                
        # 4. 套用優化後的門檻
        _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        
        x = np.arange(256, dtype=np.uint8)
        _, lut = cv2.threshold(x, thresh, 255, cv2.THRESH_BINARY) 
        self._draw_transfer_curve(lut, f"Step 4: 邊緣輔助轉換曲線 (T={thresh})")
        
        self.steps.append((f"Step 5: 邊緣輔助二值化結果", result))
        return result
