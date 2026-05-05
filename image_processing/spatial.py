# 負責空間域濾波與邊緣偵測

import cv2
import numpy as np

class SpatialMixin:
    
    # ==========================================
    # 🎨 輔助工具：計算並顯示被濾除的細節 (Difference)
    # ==========================================
    def _show_removed_noise(self, blurred_img, title="被濾除的細節 (Removed Details)"):
        """將原圖減去模糊後的圖，顯示被濾波器拿掉的細節"""
        # 為了讓暗部細節可見，加上 128 的偏移量
        diff = cv2.addWeighted(self.img, 1.0, blurred_img, -1.0, 128)
        self.steps.append((title, diff))

    # ==========================================
    # ⚙️ 平滑濾波器 (加入被濾除細節的展示)
    # ==========================================
    def mean_filter(self, kernel_size=3):
        result = cv2.blur(self.img, (kernel_size, kernel_size))
        self._show_removed_noise(result)
        return result

    def gaussian_filter(self, kernel_size=3, sigma=1.0):
        result = cv2.GaussianBlur(self.img, (kernel_size, kernel_size), sigma)
        self._show_removed_noise(result)
        return result

    def median_filter(self, kernel_size=3):
        result = cv2.medianBlur(self.img, kernel_size)
        self._show_removed_noise(result)
        return result

    def bilateral_filter(self, d=9, sigma_color=75.0, sigma_space=75.0):
        result = cv2.bilateralFilter(self.img, d, sigma_color, sigma_space)
        self._show_removed_noise(result, "被濾除的細節 (注意邊緣被保留了)")
        return result

    # ==========================================
    # ⚙️ 銳化濾波器 (展示高頻遮罩)
    # ==========================================
    def sharpen_filter(self, amount=1.0):
        # 1. 提取高頻細節 (Laplacian 概念)
        laplacian_kernel = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        details = cv2.filter2D(self.img, -1, laplacian_kernel)
        
        # 為了視覺化，將細節加上 128 偏移量
        details_vis = cv2.addWeighted(details, 1.0, np.zeros_like(details), 0, 128)
        self.steps.append(("Step 1: 提取高頻細節 (High-pass Details)", details_vis))

        # 2. 實際銳化運算
        sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5 + amount, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        sharpen_kernel = sharpen_kernel / (1 + amount)
        return cv2.filter2D(self.img, -1, sharpen_kernel)

    # ==========================================
    # ⚙️ 邊緣偵測 (展示 X 與 Y 軸梯度)
    # ==========================================
    def roberts_filter(self, direction='both'):
        gray = self._get_gray()
        x = cv2.filter2D(gray, cv2.CV_64F, np.array([[1, 0], [0, -1]], dtype=np.float32))
        y = cv2.filter2D(gray, cv2.CV_64F, np.array([[0, 1], [-1, 0]], dtype=np.float32))
        
        abs_x = cv2.convertScaleAbs(x)
        abs_y = cv2.convertScaleAbs(y)
        
        if direction == 'both':
            self.steps.append(("Step 1: X 軸梯度 (Gx)", abs_x))
            self.steps.append(("Step 2: Y 軸梯度 (Gy)", abs_y))
            return np.uint8(cv2.normalize(cv2.magnitude(x, y), None, 0, 255, cv2.NORM_MINMAX))
            
        if direction == 'x': return abs_x
        if direction == 'y': return abs_y

    def prewitt_filter(self, direction='both'):
        gray = self._get_gray()
        x = cv2.filter2D(gray, cv2.CV_64F, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32))
        y = cv2.filter2D(gray, cv2.CV_64F, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32))
        
        abs_x = cv2.convertScaleAbs(x)
        abs_y = cv2.convertScaleAbs(y)
        
        if direction == 'both':
            self.steps.append(("Step 1: X 軸梯度 (Gx)", abs_x))
            self.steps.append(("Step 2: Y 軸梯度 (Gy)", abs_y))
            return np.uint8(cv2.normalize(cv2.magnitude(x, y), None, 0, 255, cv2.NORM_MINMAX))
            
        if direction == 'x': return abs_x
        if direction == 'y': return abs_y

    def sobel_filter(self, kernel_size=3, direction='both'):
        gray = self._get_gray()
        x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        
        abs_x = cv2.convertScaleAbs(x)
        abs_y = cv2.convertScaleAbs(y)
        
        if direction == 'both':
            self.steps.append(("Step 1: 垂直邊緣 (X 軸梯度)", abs_x))
            self.steps.append(("Step 2: 水平邊緣 (Y 軸梯度)", abs_y))
            return np.uint8(cv2.normalize(cv2.magnitude(x, y), None, 0, 255, cv2.NORM_MINMAX))
            
        if direction == 'x': return abs_x
        if direction == 'y': return abs_y

    def laplacian_filter(self, kernel_size=3):
        gray = self._get_gray()
        self.steps.append(("Step 1: 轉換為灰階 (Grayscale)", gray))
        return cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size))

    def log_filter(self, kernel_size=3, sigma=1.0):
        gray = self._get_gray()
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
        self.steps.append(("Step 1: 高斯模糊 (Gaussian Blur)", blurred))
        
        laplacian = cv2.convertScaleAbs(cv2.Laplacian(blurred, cv2.CV_64F))
        self.steps.append(("Step 2: 拉普拉斯邊緣 (Laplacian)", laplacian))
        return laplacian

    def canny_filter(self, threshold1=50, threshold2=150, blur_ksize=5, blur_sigma=1.4):
        gray = self._get_gray()
        self.steps.append(("Step 1: 轉換為灰階 (Grayscale)", gray))
        
        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)
        self.steps.append(("Step 2: 高斯模糊 (Gaussian Blur)", blurred))
        
        edges = cv2.Canny(blurred, threshold1, threshold2)
        self.steps.append(("Step 3: Canny 邊緣追蹤 (Edges)", edges))
        return edges

    def canny_custom_filter(self, threshold1=50, threshold2=150, blur_ksize=5, blur_sigma=1.4):
        # 1. 轉灰階與高斯模糊
        gray = self._get_gray()
        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)
        self.steps.append(("Step 1: 高斯模糊 (Gaussian Blur)", blurred))

        # ==========================================
        # 2. 計算梯度與角度 (Sobel)
        # ==========================================
        Gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        Gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # 【修改這裡】保持 M 為原始數值，不做正規化！
        M = np.hypot(Gx, Gy) 
        
        theta = np.arctan2(Gy, Gx)
        angle = np.rad2deg(theta)
        angle[angle < 0] += 180
        
        # 【新增這裡】專門做一個用來顯示在網頁上的版本 (壓縮到 0~255)
        M_vis = np.uint8(M / M.max() * 255)
        self.steps.append(("Step 2: 梯度大小 (Gradient Magnitude)", M_vis))

        # ==========================================
        # 3. 非最大抑制 (NMS)
        # ==========================================
        # 【修改這裡】因為 M 現在是浮點數，所以 Z 也要改成浮點數陣列
        Z = np.zeros_like(M, dtype=np.float64)
        rows, cols = M.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                q = 255
                r = 255
                a = angle[i, j]
                # 判斷 4 個主方向
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
        # 【新增這裡】同樣地，NMS 完的 Z 也要做一個視覺化版本
        Z_vis = np.uint8(Z / Z.max() * 255)
        self.steps.append(("Step 3: 非最大抑制 (NMS)", Z_vis))

        # 4. 雙閾值與 Hysteresis DFS 追蹤
        res = np.zeros_like(Z, dtype=np.uint8)
        WEAK, STRONG = 75, 255
        
        strong_i, strong_j = np.where(Z >= threshold2)
        weak_i, weak_j = np.where((Z <= threshold2) & (Z >= threshold1))
        
        res[strong_i, strong_j] = STRONG
        res[weak_i, weak_j] = WEAK
        
        # 視覺化雙閾值分類結果
        thresh_vis = np.zeros_like(Z, dtype=np.uint8)
        thresh_vis[strong_i, strong_j] = 255
        thresh_vis[weak_i, weak_j] = 100
        self.steps.append((f"Step 4a: 雙閾值分類 (T1={threshold1}, T2={threshold2})", thresh_vis))

        # DFS 追蹤
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

        res[res == WEAK] = 0
        self.steps.append(("Step 4b: Hysteresis 邊緣追蹤完成", res))
        
        return res
