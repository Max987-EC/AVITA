# 形態學影像處理模組 (Morphological Image Processing Module)
# 包含基於形狀的影像運算，如侵蝕、膨脹、開閉運算及其進階應用。

import cv2
import numpy as np

class MorphologyMixin:
    """
    形態學運算混入類別 (Morphological Processing Mixin):
    提供基礎算子、空洞填補、Hit-or-Miss 變換及邊界擷取等功能。
    """
    
    def _get_structuring_element(self, shape, ksize):
        """內部工具：根據形狀名稱產生結構元素 (Structuring Element / Kernel)"""
        if shape == 'rect':
            return cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        elif shape == 'cross':
            return cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize, ksize))
        elif shape == 'ellipse':
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        return np.ones((ksize, ksize), np.uint8)

    def _apply_morphology(self, step_name, cv_op, shape, ksize, iterations=1, custom_kernel=None):
        """通用形態學執行函數：處理灰階轉換、Kernel 建立與運算執行"""
        gray = self._get_gray()
        self.steps.append(("Step 1: 轉換為灰階 (Grayscale)", gray))
        
        # 判斷是否為使用者自訂的 Kernel
        if shape == 'custom' and custom_kernel is not None:
            kernel = np.array(custom_kernel, dtype=np.uint8)
            kernel_info = f"自訂 {kernel.shape[0]}x{kernel.shape[1]}"
        else:
            kernel = self._get_structuring_element(shape, ksize)
            kernel_info = f"{ksize}x{ksize} {shape}"
        
        # 執行指定的 OpenCV 形態學運算
        if cv_op == 'erode':
            result = cv2.erode(gray, kernel, iterations=iterations)
        elif cv_op == 'dilate':
            result = cv2.dilate(gray, kernel, iterations=iterations)
        else:
            result = cv2.morphologyEx(gray, cv_op, kernel, iterations=iterations)
            
        self.steps.append((f"Step 2: {step_name} (Kernel: {kernel_info})", result))
        return result

    # ==========================================
    # 🌟 基礎形態學算子 (Basic Operators)
    # ==========================================
    def erosion(self, shape='rect', ksize=3, iterations=1, custom_kernel=None):
        """侵蝕 (Erosion): 縮小物體，消除細小雜訊。"""
        return self._apply_morphology("侵蝕 (Erosion)", 'erode', shape, ksize, iterations, custom_kernel)

    def dilation(self, shape='rect', ksize=3, iterations=1, custom_kernel=None):
        """膨脹 (Dilation): 擴大物體，填補細小空隙。"""
        return self._apply_morphology("膨脹 (Dilation)", 'dilate', shape, ksize, iterations, custom_kernel)

    def opening(self, shape='rect', ksize=3, iterations=1, custom_kernel=None):
        """斷開 (Opening): 先侵蝕再膨脹。消除亮色雜訊，平滑輪廓。"""
        return self._apply_morphology("斷開 (Opening)", cv2.MORPH_OPEN, shape, ksize, iterations, custom_kernel)

    def closing(self, shape='rect', ksize=3, iterations=1, custom_kernel=None):
        """閉合 (Closing): 先膨脹再侵蝕。填補暗色空隙，連接鄰近物件。"""
        return self._apply_morphology("閉合 (Closing)", cv2.MORPH_CLOSE, shape, ksize, iterations, custom_kernel)

    def tophat(self, shape='rect', ksize=15, custom_kernel=None):
        """頂帽轉換 (Top-Hat): 原圖 - 斷開圖。擷取比鄰域更亮的區域。"""
        return self._apply_morphology("頂帽轉換 (Top-Hat)", cv2.MORPH_TOPHAT, shape, ksize, 1, custom_kernel)

    def blackhat(self, shape='rect', ksize=15, custom_kernel=None):
        """底帽轉換 (Black-Hat): 閉合圖 - 原圖。擷取比鄰域更暗的區域。"""
        return self._apply_morphology("底帽轉換 (Black-Hat)", cv2.MORPH_BLACKHAT, shape, ksize, 1, custom_kernel)

    def morph_gradient(self, shape='rect', ksize=3, custom_kernel=None):
        """形態學梯度 (Morphological Gradient): 膨脹圖 - 侵蝕圖。擷取物件邊界。"""
        return self._apply_morphology("形態學梯度 (Gradient)", cv2.MORPH_GRADIENT, shape, ksize, 1, custom_kernel)

    # ==========================================
    # 🚀 進階應用 (Advanced Applications)
    # ==========================================
    def hole_filling(self):
        """
        空洞填補 (Hole Filling):
        利用輪廓分析尋找物件內部的封閉空洞並將其填滿。
        """
        gray = self._get_gray()
        # 1. 自動二值化
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.steps.append(("Step 1: 自動二值化 (Auto-Threshold)", binary))
        
        # 2. 尋找外輪廓 (RETR_EXTERNAL)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 3. 繪製填滿的輪廓
        filled_img = np.zeros_like(binary)
        cv2.drawContours(filled_img, contours, -1, 255, thickness=cv2.FILLED)
        
        self.steps.append(("Step 2: 填補空洞完成 (Filled)", filled_img))
        return filled_img

    def hit_or_miss(self):
        """
        擊中或擊不中轉換 (Hit-or-Miss Transform):
        用於偵測影像中特定的形狀或結構模式。在此預設尋找「十字型」特徵。
        """
        gray = self._get_gray()
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.steps.append(("Step 1: 自動二值化 (Auto-Threshold)", binary))
        
        # 定義尋找十字結構的 Kernel: 1 表示前景, -1 表示背景
        kernel = np.array([
            [-1,  1, -1],
            [ 1,  1,  1],
            [-1,  1, -1]
        ], dtype=np.int8)
        
        # 執行轉換
        result = cv2.morphologyEx(binary, cv2.MORPH_HITMISS, kernel)
        
        # 視覺化增強：將找到的點膨脹後標記在畫面上
        dilated_result = cv2.dilate(result, np.ones((5,5), np.uint8))
        output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        output[dilated_result == 255] = [0, 0, 255] # 以紅色標記特徵位置
        
        self.steps.append(("Step 2: Hit-or-Miss 偵測結果 (紅點即為特徵)", output))
        return output
    
    def boundary_extraction(self, shape='rect', ksize=3, custom_kernel=None):
        """
        形態學邊界擷取 (Boundary Extraction):
        公式：B(A) = A - (A ⊖ B)。將原圖減去其侵蝕後的結果。
        """
        gray = self._get_gray()
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.steps.append(("Step 1: 自動二值化 (Auto-Threshold)", binary))
        
        # 執行侵蝕
        if shape == 'custom' and custom_kernel is not None:
            kernel = np.array(custom_kernel, dtype=np.uint8)
        else:
            kernel = self._get_structuring_element(shape, ksize)
            
        eroded = cv2.erode(binary, kernel, iterations=1)
        
        # 原圖減去侵蝕圖即為邊界
        boundary = cv2.subtract(binary, eroded)
        
        self.steps.append((f"Step 2: 邊界擷取結果 (原圖 - 侵蝕)", boundary))
        return boundary

    def morph_smoothing(self, shape='ellipse', ksize=5, mode='open_close', custom_kernel=None):
        """
        形態學複合平滑 (Morphological Smoothing):
        組合使用斷開與閉合運算。有效去除亮色與暗色雜訊而不影響整體輪廓。
        """
        gray = self._get_gray()
        self.steps.append(("Step 1: 原始影像", gray))
        
        if shape == 'custom' and custom_kernel is not None:
            kernel = np.array(custom_kernel, dtype=np.uint8)
        else:
            kernel = self._get_structuring_element(shape, ksize)
        
        if mode == 'open_close':
            # 先 Open (去亮雜訊) 再 Close (填暗孔洞)
            opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            self.steps.append((f"Step 2: 斷開處理 (去亮雜訊)", opened))
            result = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            self.steps.append((f"Step 3: 最終平滑結果 (再執行閉合)", result))
        else:
            # 先 Close (填暗孔洞) 再 Open (去亮雜訊)
            closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            self.steps.append((f"Step 2: 閉合處理 (填暗孔洞)", closed))
            result = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            self.steps.append((f"Step 3: 最終平滑結果 (再執行斷開)", result))
            
        return result
