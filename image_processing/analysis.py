import cv2
import numpy as np

class AnalysisMixin:
    """
    分析與量測模組 (Analysis & Measurement):
    提供連通區域標記、幾何特徵擷取及次像素精度量測。
    """
    
    def connected_components(self, min_area=0, max_area=9999999, connectivity=8):
        """
        連通區域分析 (Connected Components Labeling):
        標記二值影像中的獨立物件，並根據面積進行篩選。
        
        參數:
        - min_area: 最小面積門檻，過濾雜訊。
        - max_area: 最大面積門檻。
        - connectivity: 連通性 (4 或 8 鄰域)。
        """
        gray = self._get_gray()
        # 1. 執行自動二值化 (使用 Otsu 門檻)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.steps.append(("Step 1: 自動二值化 (Auto-Threshold)", binary))
        
        # 2. 執行連通區域標記並取得統計數據
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
        
        # 建立彩色畫布用於標記結果
        output_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        valid_count = 0
        for i in range(1, num_labels): # 從 1 開始跳過背景 (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            # 面積篩選邏輯
            if min_area <= area <= max_area:
                valid_count += 1
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                (cx, cy) = centroids[i]
                
                # 繪製視覺化元件：綠色包圍盒、紅色重心與面積標註
                cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(output_img, (int(cx), int(cy)), 3, (0, 0, 255), -1)
                cv2.putText(output_img, f"#{valid_count} A:{area}", (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            
        self.steps.append((f"Step 2: 標記結果 ({connectivity}-連通, 共找到 {valid_count} 個物件)", output_img))
        return output_img

    def advanced_features(self, min_area=100, draw_bounding_rect=True, draw_min_rect=True, draw_circle=False, draw_ellipse=True, draw_hull=False, calc_gray=True, calc_perimeter=True, calc_shape=True):
        """
        進階幾何特徵擷取 (Advanced Geometric Features):
        計算物件的各種形狀描述子，包括最小外接矩形、圓度、長短軸比等。
        """
        gray = self._get_gray()
        # 1. 二值化預處理
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.steps.append(("Step 1: 自動二值化 (Auto-Threshold)", binary))
        
        # 2. 搜尋輪廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_img = self.img.copy()
        
        valid_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area: # 過濾過小區域
                continue
            valid_count += 1
            text_lines = []
            
            # 0. 繪製正交包圍盒 (Bounding Box) - 綠色
            if draw_bounding_rect:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
                if calc_shape and h > 0:
                    text_lines.append(f"B-AR:{w/h:.2f}") # 寬高比

            # 1. 繪製最小旋轉包圍盒 (Min-Area Rectangle) - 藍色
            if draw_min_rect:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(output_img, [box], 0, (255, 0, 0), 2)
                
            # 2. 繪製最小包圍圓 (Min-Enclosing Circle) - 黃色
            if draw_circle:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(output_img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                
            # 3. 繪製凸包 (Convex Hull) - 紫色
            if draw_hull:
                hull = cv2.convexHull(cnt)
                cv2.drawContours(output_img, [hull], 0, (255, 0, 255), 2)
                
            # 計算周長 (Perimeter)
            perimeter = cv2.arcLength(cnt, True)
            if calc_perimeter:
                text_lines.append(f"P:{perimeter:.1f}")
                
            # 4. 橢圓擬合與長短軸比 (Ellipse Fitting)
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                if draw_ellipse:
                    cv2.ellipse(output_img, ellipse, (0, 165, 255), 2)
                if calc_shape:
                    (center, axes, orientation) = ellipse
                    major_axis = max(axes[0], axes[1])
                    minor_axis = min(axes[0], axes[1])
                    if minor_axis > 0:
                        aspect_ratio = major_axis / minor_axis
                        text_lines.append(f"E-AR:{aspect_ratio:.2f}")
                        
            # 5. 圓度計算 (Circularity) = 4*pi*Area / Perimeter^2
            if calc_shape and perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter * perimeter)
                text_lines.append(f"C:{circularity:.2f}")
                
            # 6. 計算區域灰階特徵 (Mean/Std Dev)
            if calc_gray:
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [cnt], 0, 255, -1)
                mean_val, std_val = cv2.meanStdDev(gray, mask=mask)
                text_lines.append(f"M:{mean_val[0][0]:.1f} S:{std_val[0][0]:.1f}")
                
            # 顯示量測數據於物件旁
            if text_lines:
                text = " | ".join(text_lines)
                cv2.putText(output_img, text, (int(cnt[0][0][0]), int(cnt[0][0][1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        self.steps.append((f"Step 2: 進階特徵擷取 (共 {valid_count} 個物件)", output_img))
        return output_img

    def fuzzy_measurement(self, min_val=40, max_val=120):
        """
        模糊成員函數測量 (Fuzzy Membership Value, FMV):
        計算物件的模糊面積與模糊重心，解決邊界部分重疊或量化誤差問題。
        """
        gray = self._get_gray()
        self.steps.append(("Step 1: 原始灰階影像", gray))
        
        # 防呆：確保範圍合法
        if max_val <= min_val:
            max_val = min_val + 1
            
        # 1. 計算 FMV 權重圖 (0.0 ~ 1.0) - 分為背景、過渡、前景
        fmv = np.zeros_like(gray, dtype=np.float32)
        fmv[gray >= max_val] = 1.0 # 完全前景
        mask_mid = (gray > min_val) & (gray < max_val)
        
        # 過渡區域採線性插值
        fmv[mask_mid] = (gray[mask_mid].astype(np.float32) - min_val) / (max_val - min_val)
        
        # 視覺化：轉為熱力圖風格
        fmv_vis = (fmv * 255).astype(np.uint8)
        fmv_color = cv2.applyColorMap(fmv_vis, cv2.COLORMAP_JET)
        self.steps.append((f"Step 2: FMV 模糊權重熱力圖 ({min_val}~{max_val})", fmv_color))
        
        # 2. 定位候選物件並計算模糊面積 (Fuzzy Area)
        _, binary = cv2.threshold(gray, min_val, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        output_img = self.img.copy()
        valid_count = 0
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 10:  # 過濾雜訊
                continue
            valid_count += 1
            
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            # 擷取該物件在 FMV 權重圖中的部分
            obj_mask = (labels[y:y+h, x:x+w] == i)
            obj_fmv = fmv[y:y+h, x:x+w] * obj_mask
            
            # 積分計算模糊面積
            fuzzy_area = np.sum(obj_fmv)
            
            # 計算模糊重心 (Fuzzy Centroid) - 以權重進行座標加權平均
            if fuzzy_area > 0:
                X, Y = np.meshgrid(np.arange(w), np.arange(h))
                cx = np.sum(X * obj_fmv) / fuzzy_area + x
                cy = np.sum(Y * obj_fmv) / fuzzy_area + y
            else:
                cx, cy = centroids[i]
                
            # 繪製量測結果
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.drawMarker(output_img, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
            cv2.putText(output_img, f"FArea:{fuzzy_area:.1f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
        self.steps.append((f"Step 3: 模糊測量結果 (共 {valid_count} 個)", output_img))
        return output_img

    def subpixel_corners(self, max_corners=100, quality=0.01, min_dist=10):
        """
        次像素角點偵測 (Subpixel Corner Detection):
        先使用 Shi-Tomasi 偵測角點，再透過迭代運算優化至次像素精度。
        """
        gray = self._get_gray()
        self.steps.append(("Step 1: 轉換為灰階 (Grayscale)", gray))
        
        # 1. 取得初步的像素級角點位置
        corners = cv2.goodFeaturesToTrack(gray, max_corners, quality, min_dist)
        output_img = self.img.copy()
        
        if corners is not None:
            # 2. 定義次像素搜尋的終止準則 (criteria)
            # TERM_CRITERIA_EPS: 精度達到 0.001 時停止
            # TERM_CRITERIA_MAX_ITER: 迭代次數達到 100 次時停止
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            
            # 3. 搜尋附近的精確位置
            corners_subpix = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            
            for i in range(corners_subpix.shape[0]):
                x, y = corners_subpix[i, 0]
                # 繪製精確的紅色十字標記
                cv2.drawMarker(output_img, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
                
        self.steps.append(("Step 2: 次像素角點偵測結果", output_img))
        return output_img

    def subpixel_contours(self, threshold=127, upscale_factor=4):
        """
        次像素輪廓擷取 (Subpixel Contours via Interpolation):
        透過雙線性內插放大影像，在高解析空間擷取輪廓，再映射回原空間以獲得小數點精度。
        """
        gray = self._get_gray()
        self.steps.append(("Step 1: 原始灰階影像", gray))
        
        # 1. 使用雙線性內插 (Bilinear Interpolation) 放大影像
        # 這能模擬出原始離散像素之間的連續變化。
        h, w = gray.shape
        upscaled = cv2.resize(gray, (w * upscale_factor, h * upscale_factor), interpolation=cv2.INTER_LINEAR)
        self.steps.append((f"Step 2: 雙線性內插放大 ({upscale_factor}x)", upscaled))
        
        # 2. 在放大空間進行門檻化與輪廓擷取
        _, binary = cv2.threshold(upscaled, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # 3. 座標縮放回原尺寸，並利用 OpenCV 的次像素渲染技術 (shift 參數) 繪製
        output_img = self.img.copy()
        subpixel_count = 0
        
        for cnt in contours:
            if len(cnt) < 5:
                continue
            subpixel_count += 1
            
            # 轉換為原解析度下的小數座標
            sub_cnt = cnt.astype(np.float32) / upscale_factor
            
            # 定義位移參數 (shift=8 代表座標放大 256 倍交由底層抗鋸齒繪製)
            shift = 8
            multiplier = 2 ** shift
            # 計算放大後的整數座標
            shifted_cnt = np.round(sub_cnt * multiplier).astype(np.int32)
            
            # 使用 OpenCV 的次像素渲染技術繪製
            # 參數順序：影像, 輪廓清單, 輪廓索引, 顏色, 粗細, 線條類型, 階層, 位移
            cv2.drawContours(output_img, [shifted_cnt], 0, (0, 255, 0), 1, cv2.LINE_AA, None, shift)
            
        self.steps.append((f"Step 3: 次像素輪廓擷取 (共 {subpixel_count} 個)", output_img))
        return output_img
