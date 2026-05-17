import cv2
import numpy as np

class AnalysisMixin:
    
    def connected_components(self, min_area=0, max_area=9999999, connectivity=8):
        """連通區域標記與面積篩選 (支援 4-Neighbor 與 8-Neighbor 切換)"""
        gray = self._get_gray()
        # 確保影像是二值化狀態
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.steps.append(("Step 1: 自動二值化", binary))
        
        # 執行連通區域分析，根據參數決定連通性
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
        
        # 建立彩色畫布來繪製結果
        output_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        valid_count = 0
        for i in range(1, num_labels): # 跳過背景 (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            # 面積篩選
            if min_area <= area <= max_area:
                valid_count += 1
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                (cx, cy) = centroids[i]
                
                # 畫出綠色包圍盒與紅色重心
                cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(output_img, (int(cx), int(cy)), 3, (0, 0, 255), -1)
                cv2.putText(output_img, f"#{valid_count} A:{area}", (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            
        self.steps.append((f"Step 2: 標記結果 ({connectivity}-連通, 共找到 {valid_count} 個物件)", output_img))
        return output_img

    def advanced_features(self, min_area=100, draw_bounding_rect=True, draw_min_rect=True, draw_circle=False, draw_ellipse=True, draw_hull=False, calc_gray=True, calc_perimeter=True, calc_shape=True):
        """進階幾何特徵與灰階特徵擷取 (包含正交包圍盒、圓度與橢圓長短軸比)"""
        gray = self._get_gray()
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.steps.append(("Step 1: 自動二值化", binary))
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_img = self.img.copy()
        
        valid_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            valid_count += 1
            text_lines = []
            
            # 0. 正交包圍盒 (Axis-Parallel Bounding Box - 綠色)
            if draw_bounding_rect:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
                if calc_shape and h > 0:
                    text_lines.append(f"B-AR:{w/h:.2f}")

            # 1. 最小旋轉包圍盒 (藍色)
            if draw_min_rect:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(output_img, [box], 0, (255, 0, 0), 2)
                
            # 2. 最小包圍圓 (黃色)
            if draw_circle:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(output_img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                
            # 3. 凸包 (紫色)
            if draw_hull:
                hull = cv2.convexHull(cnt)
                cv2.drawContours(output_img, [hull], 0, (255, 0, 255), 2)
                
            # 計算周長
            perimeter = cv2.arcLength(cnt, True)
            if calc_perimeter:
                text_lines.append(f"P:{perimeter:.1f}")
                
            # 4. 橢圓擬合與長短軸比 (Aspect Ratio)
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
                
            # 6. 計算區域灰階
            if calc_gray:
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [cnt], 0, 255, -1)
                mean_val, std_val = cv2.meanStdDev(gray, mask=mask)
                text_lines.append(f"M:{mean_val[0][0]:.1f} S:{std_val[0][0]:.1f}")
                
            # 將數據寫在物件旁邊
            if text_lines:
                text = " | ".join(text_lines)
                cv2.putText(output_img, text, (int(cnt[0][0][0]), int(cnt[0][0][1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        self.steps.append((f"Step 2: 進階特徵擷取 (共 {valid_count} 個物件)", output_img))
        return output_img

    def fuzzy_measurement(self, min_val=40, max_val=120):
        """模糊成員函數測量 (Fuzzy Membership Value, FMV) - 簡報 P.178-181"""
        gray = self._get_gray()
        self.steps.append(("Step 1: 原始灰階影像", gray))
        
        # [新增] 防呆機制：確保 max_val 永遠大於 min_val，避免除以零
        if max_val <= min_val:
            max_val = min_val + 1
            
        # 1. 計算 FMV 權重圖 (0.0 ~ 1.0)
        fmv = np.zeros_like(gray, dtype=np.float32)
        fmv[gray >= max_val] = 1.0
        mask_mid = (gray > min_val) & (gray < max_val)
        
        # [優化] 將 uint8 強制轉為 float32 再相減，確保數值安全不溢位
        fmv[mask_mid] = (gray[mask_mid].astype(np.float32) - min_val) / (max_val - min_val)
        
        # 將 FMV 轉為熱力圖視覺化
        fmv_vis = (fmv * 255).astype(np.uint8)
        fmv_color = cv2.applyColorMap(fmv_vis, cv2.COLORMAP_JET)
        self.steps.append((f"Step 2: FMV 模糊權重熱力圖 ({min_val}~{max_val})", fmv_color))
        
        # 2. 找出候選區域並進行次像素模糊測量
        _, binary = cv2.threshold(gray, min_val, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        output_img = self.img.copy()
        valid_count = 0
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 10:  # 過濾極小雜訊
                continue
            valid_count += 1
            
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            # 擷取該物件的 FMV 權重
            obj_mask = (labels[y:y+h, x:x+w] == i)
            obj_fmv = fmv[y:y+h, x:x+w] * obj_mask
            
            # 計算模糊面積 (Fuzzy Area)
            fuzzy_area = np.sum(obj_fmv)
            
            # 計算模糊重心 (Fuzzy Centroid)
            if fuzzy_area > 0:
                X, Y = np.meshgrid(np.arange(w), np.arange(h))
                cx = np.sum(X * obj_fmv) / fuzzy_area + x
                cy = np.sum(Y * obj_fmv) / fuzzy_area + y
            else:
                cx, cy = centroids[i]
                
            # 繪製結果
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.drawMarker(output_img, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
            cv2.putText(output_img, f"FArea:{fuzzy_area:.1f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
        self.steps.append((f"Step 3: 模糊測量結果 (共 {valid_count} 個)", output_img))
        return output_img

    def subpixel_corners(self, max_corners=100, quality=0.01, min_dist=10):
        """次像素精度門檻化 (Subpixel Corners)"""
        gray = self._get_gray()
        
        # [新增] 讓 UI 顯示第一步是灰階圖
        self.steps.append(("Step 1: 轉換為灰階 (Grayscale)", gray))
        
        # 1. 先找出像素級的角點
        corners = cv2.goodFeaturesToTrack(gray, max_corners, quality, min_dist)
        output_img = self.img.copy()
        
        if corners is not None:
            # 2. 定義次像素搜尋參數 (迭代次數與精度)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            # 3. 計算次像素精確位置
            corners_subpix = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            
            for i in range(corners_subpix.shape[0]):
                x, y = corners_subpix[i, 0]
                # 畫出紅色的精確十字標記
                cv2.drawMarker(output_img, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
                
        # [修正] 補上 Step 2 確保 UI 一致性
        self.steps.append(("Step 2: 次像素角點偵測結果", output_img))
        return output_img

    def subpixel_contours(self, threshold=127, upscale_factor=4):
        """次像素輪廓擷取 (基於雙線性內插與次像素渲染) - 簡報 P.112"""
        gray = self._get_gray()
        self.steps.append(("Step 1: 原始灰階影像", gray))
        
        # 1. 雙線性內插放大 (Bilinear Interpolation) 形成連續函數空間
        h, w = gray.shape
        upscaled = cv2.resize(gray, (w * upscale_factor, h * upscale_factor), interpolation=cv2.INTER_LINEAR)
        self.steps.append((f"Step 2: 雙線性內插放大 ({upscale_factor}x)", upscaled))
        
        # 2. 在高解析度空間進行門檻化與輪廓擷取
        _, binary = cv2.threshold(upscaled, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # 3. 縮放回原尺寸並使用 OpenCV 次像素精度繪製 (shift 參數)
        output_img = self.img.copy()
        subpixel_count = 0
        
        for cnt in contours:
            if len(cnt) < 5:
                continue
            subpixel_count += 1
            
            # 將座標除以放大倍率，得到帶有小數點的次像素座標
            sub_cnt = cnt.astype(np.float32) / upscale_factor
            
            # 使用 shift=8 進行次像素渲染 (座標放大 256 倍交由底層抗鋸齒繪製)
            shift = 8
            multiplier = 2 ** shift
            shifted_cnt = np.round(sub_cnt * multiplier).astype(np.int32)
            
            # 繪製高精度抗鋸齒綠色輪廓
            cv2.drawContours(output_img, [shifted_cnt], 0, (0, 255, 0), 1, cv2.LINE_AA, shift=shift)
            
        self.steps.append((f"Step 3: 次像素輪廓擷取 (共 {subpixel_count} 個)", output_img))
        return output_img
