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
