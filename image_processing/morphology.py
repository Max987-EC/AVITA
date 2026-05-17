# 負責形態學運算 (Morphological Processing)

import cv2
import numpy as np

class MorphologyMixin:
    
    def _get_structuring_element(self, shape, ksize):
        """產生指定形狀與大小的結構元素"""
        if shape == 'rect':
            return cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        elif shape == 'cross':
            return cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize, ksize))
        elif shape == 'ellipse':
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        return np.ones((ksize, ksize), np.uint8)

    def _apply_morphology(self, step_name, cv_op, shape, ksize, iterations=1):
        """底層共用的形態學執行函數"""
        gray = self._get_gray()
        self.steps.append(("Step 1: 轉換為灰階 (Grayscale)", gray))
        
        kernel = self._get_structuring_element(shape, ksize)
        
        if cv_op == 'erode':
            result = cv2.erode(gray, kernel, iterations=iterations)
        elif cv_op == 'dilate':
            result = cv2.dilate(gray, kernel, iterations=iterations)
        else:
            result = cv2.morphologyEx(gray, cv_op, kernel, iterations=iterations)
            
        self.steps.append((f"Step 2: {step_name} (Kernel: {ksize}x{ksize} {shape})", result))
        return result

    # ==========================================
    # 🌟 拆分後的 7 個獨立形態學算子
    # ==========================================
    def erosion(self, shape='rect', ksize=3, iterations=1):
        return self._apply_morphology("侵蝕 (Erosion)", 'erode', shape, ksize, iterations)

    def dilation(self, shape='rect', ksize=3, iterations=1):
        return self._apply_morphology("膨脹 (Dilation)", 'dilate', shape, ksize, iterations)

    def opening(self, shape='rect', ksize=3, iterations=1):
        return self._apply_morphology("斷開 (Opening)", cv2.MORPH_OPEN, shape, ksize, iterations)

    def closing(self, shape='rect', ksize=3, iterations=1):
        return self._apply_morphology("閉合 (Closing)", cv2.MORPH_CLOSE, shape, ksize, iterations)

    def tophat(self, shape='rect', ksize=15):
        return self._apply_morphology("頂帽轉換 (Top-Hat)", cv2.MORPH_TOPHAT, shape, ksize)

    def blackhat(self, shape='rect', ksize=15):
        return self._apply_morphology("底帽轉換 (Black-Hat)", cv2.MORPH_BLACKHAT, shape, ksize)

    def morph_gradient(self, shape='rect', ksize=3):
        return self._apply_morphology("形態學梯度 (Gradient)", cv2.MORPH_GRADIENT, shape, ksize)

    def hole_filling(self):
        """空洞填補 (Hole Filling)"""
        gray = self._get_gray()
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.steps.append(("Step 1: 自動二值化", binary))
        
        # 尋找輪廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 建立全黑畫布，將輪廓內部填滿白色
        filled_img = np.zeros_like(binary)
        cv2.drawContours(filled_img, contours, -1, 255, thickness=cv2.FILLED)
        
        self.steps.append(("Step 2: 填補空洞", filled_img))
        return filled_img

    def hit_or_miss(self):
        """交離轉換 (Hit-or-Miss) - 尋找十字交叉點特徵"""
        gray = self._get_gray()
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.steps.append(("Step 1: 自動二值化", binary))
        
        # 定義一個尋找「十字型」特徵的 Kernel (1: 前景, -1: 背景, 0: 不在乎)
        kernel = np.array([
            [-1,  1, -1],
            [ 1,  1,  1],
            [-1,  1, -1]
        ], dtype=np.int8)
        
        result = cv2.morphologyEx(binary, cv2.MORPH_HITMISS, kernel)
        
        # 為了讓結果明顯，我們將找到的點膨脹並標示在原圖上
        dilated_result = cv2.dilate(result, np.ones((5,5), np.uint8))
        output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        output[dilated_result == 255] = [0, 0, 255] # 將找到的特徵標為紅色
        
        self.steps.append(("Step 2: 交離轉換結果 (紅點為特徵位置)", output))
        return output
    
    def boundary_extraction(self, shape='rect', ksize=3):
        """形態學邊界抽取: 原圖 - 侵蝕圖 (簡報 P.148)"""
        gray = self._get_gray()
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.steps.append(("Step 1: 自動二值化", binary))
        
        # 取得結構元素並進行侵蝕
        kernel = self._get_structuring_element(shape, ksize)
        eroded = cv2.erode(binary, kernel, iterations=1)
        
        # 邊界 = 原圖 - 侵蝕圖
        boundary = cv2.subtract(binary, eroded)
        
        self.steps.append((f"Step 2: 邊界抽取 (原圖減去侵蝕, Kernel: {ksize}x{ksize})", boundary))
        return boundary

    def morph_smoothing(self, shape='ellipse', ksize=5, mode='open_close'):
        """形態學複合平滑 (Open-Close 或 Close-Open)"""
        gray = self._get_gray()
        self.steps.append(("Step 1: 原始影像", gray))
        
        kernel = self._get_structuring_element(shape, ksize)
        
        if mode == 'open_close':
            # 先斷開 (去除亮雜訊) 再閉合 (填補暗孔洞)
            opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            self.steps.append((f"Step 2: 斷開處理 (去除亮雜訊)", opened))
            result = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            self.steps.append((f"Step 3: 閉合處理 (填補暗孔洞) - 最終結果", result))
        else:
            # 先閉合 (填補暗孔洞) 再斷開 (去除亮雜訊)
            closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            self.steps.append((f"Step 2: 閉合處理 (填補暗孔洞)", closed))
            result = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            self.steps.append((f"Step 3: 斷開處理 (去除亮雜訊) - 最終結果", result))
            
        return result
