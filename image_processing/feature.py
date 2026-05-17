# 特徵檢測模組 (Feature Detection Module)
# 包含霍夫變換 (Hough Transform) 相關的直線與圓形偵測演算法。

import cv2
import numpy as np

class FeatureMixin:
    """
    特徵檢測混入類別 (Feature Detection Mixin):
    提供偵測影像中幾何形狀 (線、圓) 的功能，並包含霍夫空間視覺化。
    """
    
    # ==========================================
    # 🎨 輔助工具：繪製霍夫空間累加器 (Hough Space Visualization)
    # ==========================================
    def _draw_hough_space(self, edges, rho_res, theta_deg_res):
        """
        將邊緣影像轉換為霍夫空間的投票圖 (Accumulator Map):
        幫助理解直線偵測的原理，即「影像空間中的一個點，在霍夫空間中是一條正弦曲線」。
        """
        # 為加速計算並避免網頁載入過久，將影像縮小處理
        small_edges = cv2.resize(edges, (150, 150))
        y_idxs, x_idxs = np.nonzero(small_edges)
        
        # 定義角度範圍 theta (-90 到 90 度)
        thetas = np.deg2rad(np.arange(-90, 90, max(1.0, theta_deg_res)))
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        
        # 計算最大可能的 rho 值 (影像對角線長度)
        diag_len = int(np.ceil(np.sqrt(150**2 + 150**2)))
        accumulator = np.zeros((2 * diag_len, len(thetas)), dtype=np.float64)
        
        # 進行投票 (Voting): 遍歷所有邊緣點
        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]
            # 計算該點在各角度下對應的 rho 值: rho = x*cos(theta) + y*sin(theta)
            rhos = np.round(x * cos_t + y * sin_t).astype(int)
            for t_idx, rho in enumerate(rhos):
                accumulator[rho + diag_len, t_idx] += 1
                
        # 視覺化：將累加器數據轉為熱力圖顏色顯示
        acc_vis = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        acc_color = cv2.applyColorMap(acc_vis, cv2.COLORMAP_HOT)
        self.steps.append(("Step 4: 霍夫空間投票圖 (Hough Accumulator)", acc_color))

    # ==========================================
    # ⚙️ 霍夫直線偵測 (Hough Line Transform)
    # ==========================================
    def hough_lines_standard(self, threshold=150, rho=1.0, theta_deg=1.0, canny_th1=50, canny_th2=150, blur_ksize=5):
        """
        標準霍夫直線偵測 (Standard Hough Transform):
        偵測通過全圖的無限長直線。返回 (rho, theta) 參數化的線條。
        """
        gray = self._get_gray()
        self.steps.append(("Step 1: 轉換為灰階 (Grayscale)", gray))
        
        # 1. 預處理：高斯模糊去噪
        if blur_ksize % 2 == 0: blur_ksize += 1
        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        self.steps.append(("Step 2: 高斯模糊去噪 (Blurred)", blurred))
        
        # 2. 邊緣檢測：Canny 算子
        edges = cv2.Canny(blurred, canny_th1, canny_th2)
        self.steps.append(("Step 3: Canny 邊緣影像 (Edges)", edges))
        
        # 3. 視覺化霍夫空間累加器
        self._draw_hough_space(edges, rho, theta_deg)
        
        # 4. 執行 OpenCV 標準霍夫轉換
        theta = theta_deg * np.pi / 180.0
        lines = cv2.HoughLines(edges, rho, theta, threshold)
        
        result = self.img.copy() if self.is_color else cv2.cvtColor(self.img.copy(), cv2.COLOR_GRAY2BGR)
            
        # 將 (rho, theta) 轉換回笛卡兒座標系並繪製直線
        if lines is not None:
            for line in lines:
                r, t = line[0]
                a, b = np.cos(t), np.sin(t)
                x0, y0 = a * r, b * r
                # 計算兩個遠端點以繪製無限延伸的線
                x1 = int(x0 + 10000 * (-b))
                y1 = int(y0 + 10000 * (a))
                x2 = int(x0 - 10000 * (-b))
                y2 = int(y0 - 10000 * (a))
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
        self.steps.append(("Step 5: 繪製無限長直線 (Standard Lines)", result))
        return result

    def hough_lines_p(self, threshold=100, min_line_length=50, max_line_gap=10, rho=1.0, theta_deg=1.0, canny_th1=50, canny_th2=150, blur_ksize=5):
        """
        漸進式機率霍夫直線偵測 (Probabilistic Hough Transform):
        僅對部分點進行投票，運算較快且能偵測出具備起始點與結束點的「線段」。
        """
        gray = self._get_gray()
        self.steps.append(("Step 1: 轉換為灰階 (Grayscale)", gray))
        
        # 1. 預處理
        if blur_ksize % 2 == 0: blur_ksize += 1
        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        self.steps.append(("Step 2: 高斯模糊去噪 (Blurred)", blurred))
        
        # 2. 邊緣檢測
        edges = cv2.Canny(blurred, canny_th1, canny_th2)
        self.steps.append(("Step 3: Canny 邊緣影像 (Edges)", edges))
        
        # 3. 執行漸進式霍夫轉換
        theta = theta_deg * np.pi / 180.0
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, 
                                minLineLength=min_line_length, maxLineGap=max_line_gap)
        
        result = self.img.copy() if self.is_color else cv2.cvtColor(self.img.copy(), cv2.COLOR_GRAY2BGR)
            
        # 繪製線段
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        self.steps.append(("Step 4: 繪製線段結果 (Probabilistic Lines)", result))
        return result

    # ==========================================
    # ⚙️ 霍夫圓形偵測 (Hough Circle Transform)
    # ==========================================
    def hough_circles(self, dp=1.0, min_dist=20, param1=50, param2=30, min_radius=0, max_radius=0, blur_ksize=5):
        """
        霍夫圓形偵測 (Hough Gradient Method):
        利用邊緣梯度方向減少投票空間。此方法不需要預先執行 Canny，其內部會自動處理。
        
        參數:
        - dp: 累加器解析度比例 (1 表示與原圖相同)。
        - min_dist: 圓心間的最小距離。
        - param1: Canny 邊緣檢測的高門檻值。
        - param2: 累加器門檻 (投票數)。
        """
        gray = self._get_gray()
        self.steps.append(("Step 1: 轉換為灰階 (Grayscale)", gray))
        
        # 1. 預處理：使用中值模糊對於圓形偵測通常效果較佳 (去除椒鹽雜訊)
        blurred = cv2.medianBlur(gray, blur_ksize)
        self.steps.append(("Step 2: 中值模糊去噪 (Median Blur)", blurred))
        
        # 2. 視覺化：模擬 HoughCircles 內部隱藏的 Canny 邊緣步驟
        # (低門檻通常被設定為高門檻的一半)
        hidden_edges = cv2.Canny(blurred, param1 // 2, param1)
        self.steps.append(("Step 3: 內部 Canny 邊緣 (由 Param1 控制)", hidden_edges))
        
        # 3. 執行圓形偵測
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp, min_dist,
                                   param1=param1, param2=param2, 
                                   minRadius=min_radius, maxRadius=max_radius)
        
        result = self.img.copy() if self.is_color else cv2.cvtColor(self.img.copy(), cv2.COLOR_GRAY2BGR)
            
        # 繪製圓形與圓心
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # 繪製圓周 (綠色)
                cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # 繪製圓心 (紅色)
                cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)
                
        self.steps.append(("Step 4: 繪製圓形與圓心結果 (Draw Circles)", result))
        return result
