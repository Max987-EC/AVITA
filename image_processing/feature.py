# 負責霍夫特徵檢測

import cv2
import numpy as np

class FeatureMixin:
    
    # ==========================================
    # 🎨 輔助工具：繪製霍夫空間累加器 (Hough Space)
    # ==========================================
    def _draw_hough_space(self, edges, rho_res, theta_deg_res):
        """將邊緣點轉換為霍夫空間的投票圖 (Accumulator)"""
        # 為了避免網頁卡頓，將邊緣圖縮小以加速計算
        small_edges = cv2.resize(edges, (150, 150))
        y_idxs, x_idxs = np.nonzero(small_edges)
        
        # 定義 theta 範圍 (-90 到 90 度)
        thetas = np.deg2rad(np.arange(-90, 90, max(1.0, theta_deg_res)))
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        
        # 計算最大可能的 rho 值 (對角線長度)
        diag_len = int(np.ceil(np.sqrt(150**2 + 150**2)))
        accumulator = np.zeros((2 * diag_len, len(thetas)), dtype=np.float64)
        
        # 進行投票 (Voting)
        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]
            rhos = np.round(x * cos_t + y * sin_t).astype(int)
            for t_idx, rho in enumerate(rhos):
                accumulator[rho + diag_len, t_idx] += 1
                
        # 視覺化：將累加器轉為熱力圖風格
        acc_vis = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        acc_color = cv2.applyColorMap(acc_vis, cv2.COLORMAP_HOT)
        self.steps.append(("Step 4: 霍夫空間投票圖 (Hough Accumulator)", acc_color))

    # ==========================================
    # ⚙️ 演算法實作
    # ==========================================
    def hough_lines_standard(self, threshold=150, rho=1.0, theta_deg=1.0, canny_th1=50, canny_th2=150, blur_ksize=5):
        gray = self._get_gray()
        self.steps.append(("Step 1: 轉換為灰階 (Grayscale)", gray))
        
        # 🌟 新增：高斯模糊前處理
        if blur_ksize % 2 == 0: blur_ksize += 1 # 確保 kernel size 是奇數
        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        self.steps.append(("Step 2: 高斯模糊去噪 (Blurred)", blurred))
        
        # 將 blurred 傳給 Canny，而不是 gray
        edges = cv2.Canny(blurred, canny_th1, canny_th2)
        self.steps.append(("Step 3: Canny 邊緣 (Edges)", edges))
        
        # 🎨 視覺化霍夫空間 (這非常具備教學價值！)
        self._draw_hough_space(edges, rho, theta_deg)
        
        theta = theta_deg * np.pi / 180.0
        lines = cv2.HoughLines(edges, rho, theta, threshold)
        
        result = self.img.copy() if self.is_color else cv2.cvtColor(self.img.copy(), cv2.COLOR_GRAY2BGR)
            
        if lines is not None:
            for line in lines:
                r, t = line[0]
                a = np.cos(t)
                b = np.sin(t)
                x0 = a * r
                y0 = b * r
                x1 = int(x0 + 10000 * (-b))
                y1 = int(y0 + 10000 * (a))
                x2 = int(x0 - 10000 * (-b))
                y2 = int(y0 - 10000 * (a))
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
        self.steps.append(("Step 5: 繪製無限長直線 (Standard Lines)", result))
        return result

    def hough_lines_p(self, threshold=100, min_line_length=50, max_line_gap=10, rho=1.0, theta_deg=1.0, canny_th1=50, canny_th2=150, blur_ksize=5):
        gray = self._get_gray()
        self.steps.append(("Step 1: 轉換為灰階 (Grayscale)", gray))
        
        # 🌟 新增：高斯模糊前處理
        if blur_ksize % 2 == 0: blur_ksize += 1
        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        self.steps.append(("Step 2: 高斯模糊去噪 (Blurred)", blurred))
        
        # 將 blurred 傳給 Canny
        edges = cv2.Canny(blurred, canny_th1, canny_th2)
        self.steps.append(("Step 3: Canny 邊緣 (Edges)", edges))
        
        theta = theta_deg * np.pi / 180.0
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
        
        result = self.img.copy() if self.is_color else cv2.cvtColor(self.img.copy(), cv2.COLOR_GRAY2BGR)
            
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        self.steps.append(("Step 4: 繪製線段 (Probabilistic Lines)", result))
        return result

    def hough_circles(self, dp=1.0, min_dist=20, param1=50, param2=30, min_radius=0, max_radius=0, blur_ksize=5):
        gray = self._get_gray()
        self.steps.append(("Step 1: 轉換為灰階 (Grayscale)", gray))
        
        blurred = cv2.medianBlur(gray, blur_ksize)
        self.steps.append(("Step 2: 中值模糊去噪 (Median Blur)", blurred))
        
        # 🎨 視覺化：模擬 HoughCircles 內部隱藏的 Canny 邊緣
        # HoughCircles 內部的 Canny 低門檻固定為高門檻 (param1) 的一半
        hidden_edges = cv2.Canny(blurred, param1 // 2, param1)
        self.steps.append(("Step 3: 內部 Canny 邊緣 (由 Param1 控制)", hidden_edges))
        
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp, min_dist,
                                   param1=param1, param2=param2, 
                                   minRadius=min_radius, maxRadius=max_radius)
        
        result = self.img.copy() if self.is_color else cv2.cvtColor(self.img.copy(), cv2.COLOR_GRAY2BGR)
            
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)
                
        self.steps.append(("Step 4: 繪製圓形與圓心 (Draw Circles)", result))
        return result
