# 空間域濾波與邊緣偵測模組 (Spatial Domain Filtering & Edge Detection)
# 包含在影像像素空間內進行卷積 (Convolution) 運算的各種濾波器。

import cv2
import numpy as np

class SpatialMixin:
    """
    空間域處理混入類別 (Spatial Domain Processing Mixin):
    提供平滑 (去噪)、銳化及多種邊緣偵測算子 (Sobel, Canny, Laplacian)。
    """
    
    # ==========================================
    # 🎨 輔助工具：顯示被濾除細節 (Difference Visualization)
    # ==========================================
    def _show_removed_noise(self, blurred_img, title="Step 2: 被濾除的細節 (Removed Details)"):
        """
        將原圖減去模糊後的影像：
        視覺化該濾波器究竟「拿掉了什麼」。這對於理解低通濾波器非常有幫助。
        """
        # 加上 128 偏移量讓「正負差異」都能在灰階圖中被看見
        diff = cv2.addWeighted(self.img, 1.0, blurred_img, -1.0, 128)
        self.steps.append((title, diff))

    # ==========================================
    # ⚙️ 平滑/低通濾波器 (Smoothing Filters)
    # ==========================================
    def mean_filter(self, kernel_size=3):
        """均值濾波 (Mean Filter): 取鄰域平均值。簡單但會使邊緣模糊。"""
        result = cv2.blur(self.img, (kernel_size, kernel_size))
        self.steps.append(("Step 1: 均值平滑結果", result))
        self._show_removed_noise(result)
        return result

    def gaussian_filter(self, kernel_size=3, sigma=1.0):
        """高斯濾波 (Gaussian Filter): 權重隨距離正態分佈。比均值濾波更能保留結構。"""
        result = cv2.GaussianBlur(self.img, (kernel_size, kernel_size), sigma)
        self.steps.append(("Step 1: 高斯平滑結果", result))
        self._show_removed_noise(result)
        return result

    def median_filter(self, kernel_size=3):
        """中值濾波 (Median Filter): 取鄰域中位數。極佳的椒鹽雜訊 (Salt-and-Pepper) 去除能力。"""
        result = cv2.medianBlur(self.img, kernel_size)
        self.steps.append(("Step 1: 中值平滑結果", result))
        self._show_removed_noise(result)
        return result

    def bilateral_filter(self, d=9, sigma_color=75.0, sigma_space=75.0):
        """雙邊濾波 (Bilateral Filter): 同時考慮空間距離與色彩差異。能去噪且「保留邊緣」。"""
        result = cv2.bilateralFilter(self.img, d, sigma_color, sigma_space)
        self.steps.append(("Step 1: 雙邊平滑結果", result))
        self._show_removed_noise(result, "Step 2: 被濾除的細節 (可觀察到邊緣被完整保留)")
        return result

    # ==========================================
    # ⚙️ 銳化/高通濾波器 (Sharpening Filters)
    # ==========================================
    def sharpen_filter(self, amount=1.0):
        """
        銳化濾波 (Sharpening):
        增強影像的高頻細節 (邊緣)。
        """
        # 1. 建立拉普拉斯核來提取細節 (High-pass Details)
        laplacian_kernel = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        details = cv2.filter2D(self.img, -1, laplacian_kernel)
        
        details_vis = cv2.addWeighted(details, 1.0, np.zeros_like(details), 0, 128)
        self.steps.append(("Step 1: 提取高頻細節 (High-pass Details)", details_vis))

        # 2. 執行銳化卷積
        sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5 + amount, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        # 正規化權重以保持亮度一致
        sharpen_kernel = sharpen_kernel / (1 + amount)
        
        result = cv2.filter2D(self.img, -1, sharpen_kernel)
        self.steps.append(("Step 2: 銳化最終結果", result))
        return result

    # ==========================================
    # ⚙️ 邊緣偵測算子 (Edge Detection)
    # ==========================================
    def roberts_filter(self, direction='both'):
        """Roberts 算子: 使用 2x2 對角線差分。對雜訊敏感，適用於簡單邊緣。"""
        gray = self._get_gray()
        x = cv2.filter2D(gray, cv2.CV_64F, np.array([[1, 0], [0, -1]], dtype=np.float32))
        y = cv2.filter2D(gray, cv2.CV_64F, np.array([[0, 1], [-1, 0]], dtype=np.float32))
        
        abs_x, abs_y = cv2.convertScaleAbs(x), cv2.convertScaleAbs(y)
        
        if direction == 'both':
            self.steps.append(("Step 1: X 軸梯度 (Gx)", abs_x))
            self.steps.append(("Step 2: Y 軸梯度 (Gy)", abs_y))
            result = np.uint8(cv2.normalize(cv2.magnitude(x, y), None, 0, 255, cv2.NORM_MINMAX))
            self.steps.append(("Step 3: 綜合邊緣強度 (Magnitude)", result))
            return result
        return abs_x if direction == 'x' else abs_y

    def prewitt_filter(self, direction='both'):
        """Prewitt 算子: 3x3 差分算子。比 Roberts 穩定，適合偵測水平/垂直線條。"""
        gray = self._get_gray()
        x = cv2.filter2D(gray, cv2.CV_64F, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32))
        y = cv2.filter2D(gray, cv2.CV_64F, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32))
        
        abs_x, abs_y = cv2.convertScaleAbs(x), cv2.convertScaleAbs(y)
        
        if direction == 'both':
            self.steps.append(("Step 1: X 軸梯度 (Gx)", abs_x))
            self.steps.append(("Step 2: Y 軸梯度 (Gy)", abs_y))
            result = np.uint8(cv2.normalize(cv2.magnitude(x, y), None, 0, 255, cv2.NORM_MINMAX))
            self.steps.append(("Step 3: 綜合邊緣強度 (Magnitude)", result))
            return result
        return abs_x if direction == 'x' else abs_y

    def sobel_filter(self, kernel_size=3, direction='both'):
        """Sobel 算子: 加入高斯平滑權重的差分。目前工業界最常用的基礎邊緣檢測。"""
        gray = self._get_gray()
        x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        
        abs_x, abs_y = cv2.convertScaleAbs(x), cv2.convertScaleAbs(y)
        
        if direction == 'both':
            self.steps.append(("Step 1: 垂直邊緣 (Gx)", abs_x))
            self.steps.append(("Step 2: 水平邊緣 (Gy)", abs_y))
            result = np.uint8(cv2.normalize(cv2.magnitude(x, y), None, 0, 255, cv2.NORM_MINMAX))
            self.steps.append(("Step 3: 綜合邊緣強度 (Magnitude)", result))
            return result
        return abs_x if direction == 'x' else abs_y

    def laplacian_filter(self, kernel_size=3):
        """拉普拉斯算子 (Laplacian): 二階導數算子。用於偵測影像中的快速變化 (邊緣點)。"""
        gray = self._get_gray()
        self.steps.append(("Step 1: 轉換為灰階 (Grayscale)", gray))
        result = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size))
        self.steps.append(("Step 2: 拉普拉斯邊緣 (Laplacian)", result))
        return result

    def log_filter(self, kernel_size=3, sigma=1.0):
        """高斯拉普拉斯 (LoG): 先用高斯模糊降噪，再執行拉普拉斯。能更精準定位邊緣。"""
        gray = self._get_gray()
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
        self.steps.append(("Step 1: 高斯模糊 (Gaussian Blur)", blurred))
        
        laplacian = cv2.convertScaleAbs(cv2.Laplacian(blurred, cv2.CV_64F))
        self.steps.append(("Step 2: 對模糊後的圖執行拉普拉斯", laplacian))
        return laplacian

    def canny_filter(self, threshold1=50, threshold2=150, blur_ksize=5, blur_sigma=1.4):
        """Canny 邊緣檢測 (OpenCV 版): 包含多個步驟的最強邊緣檢測演算法。"""
        gray = self._get_gray()
        self.steps.append(("Step 1: 轉換為灰階 (Grayscale)", gray))
        
        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)
        self.steps.append(("Step 2: 高斯模糊 (Gaussian Blur)", blurred))
        
        edges = cv2.Canny(blurred, threshold1, threshold2)
        self.steps.append(("Step 3: Canny 邊緣追蹤最終結果 (Edges)", edges))
        return edges

    def canny_custom_filter(self, threshold1=50, threshold2=150, blur_ksize=5, blur_sigma=1.4):
        """
        自定義 Canny 演算法實作 (教學用):
        展示 Canny 的核心流程：高斯模糊 -> 梯度計算 -> 非最大抑制 (NMS) -> 雙門檻追蹤。
        """
        # 1. 預處理
        gray = self._get_gray()
        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)
        self.steps.append(("Step 1: 高斯模糊 (Gaussian Blur)", blurred))

        # 2. 計算梯度強度與角度 (Sobel)
        Gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        Gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        M = np.hypot(Gx, Gy) # 梯度大小
        theta = np.arctan2(Gy, Gx)
        angle = np.rad2deg(theta)
        angle[angle < 0] += 180
        
        M_vis = np.uint8(M / M.max() * 255)
        self.steps.append(("Step 2: 梯度大小圖 (Gradient Magnitude)", M_vis))

        # 3. 非最大抑制 (Non-Maximum Suppression): 變薄邊緣，只保留局部最大值。
        Z = np.zeros_like(M, dtype=np.float64)
        rows, cols = M.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                q, r = 255, 255
                a = angle[i, j]
                # 根據 4 個方向判斷鄰域像素
                if (0 <= a < 22.5) or (157.5 <= a <= 180):
                    q, r = M[i, j+1], M[i, j-1]
                elif (22.5 <= a < 67.5):
                    q, r = M[i+1, j-1], M[i-1, j+1]
                elif (67.5 <= a < 112.5):
                    q, r = M[i+1, j], M[i-1, j]
                elif (112.5 <= a < 157.5):
                    q, r = M[i-1, j-1], M[i+1, j+1]

                if (M[i, j] >= q) and (M[i, j] >= r):
                    Z[i, j] = M[i, j]
                else:
                    Z[i, j] = 0
                    
        Z_vis = np.uint8(Z / Z.max() * 255)
        self.steps.append(("Step 3: 非最大抑制 (NMS) - 邊緣變薄", Z_vis))

        # 4. 雙門檻 (Double Threshold) 與滯後追蹤 (Hysteresis)
        res = np.zeros_like(Z, dtype=np.uint8)
        WEAK, STRONG = 75, 255
        
        strong_i, strong_j = np.where(Z >= threshold2)
        weak_i, weak_j = np.where((Z <= threshold2) & (Z >= threshold1))
        
        res[strong_i, strong_j] = STRONG
        res[weak_i, weak_j] = WEAK
        
        # 視覺化分類
        thresh_vis = np.zeros_like(Z, dtype=np.uint8)
        thresh_vis[strong_i, strong_j] = 255
        thresh_vis[weak_i, weak_j] = 100
        self.steps.append((f"Step 4a: 雙門檻分類 (強/弱邊緣)", thresh_vis))

        # 滯後追蹤：若弱邊緣連接著強邊緣，則將其保留。
        stack = list(zip(strong_i, strong_j))
        while stack:
            r, c = stack.pop()
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and res[nr, nc] == WEAK:
                        res[nr, nc] = STRONG
                        stack.append((nr, nc))

        res[res == WEAK] = 0 # 丟棄未連接的弱邊緣
        self.steps.append(("Step 4b: 滯後追蹤完成 (最終 Canny 結果)", res))
        
        return res
