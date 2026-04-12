import cv2
import numpy as np

class LaneDetector:
    def __init__(self):
        # 將原本的全域變數，變成每個「物件實體」專屬的屬性
        self.prev_left_lane = None
        self.prev_right_lane = None
        self.left_miss_count = 0
        self.right_miss_count = 0

        self.prev_outer_left_lane = None
        self.prev_outer_right_lane = None
        self.outer_left_miss_count = 0
        self.outer_right_miss_count = 0

        self.MAX_MISS_FRAMES = 60

    def apply_white_balance(self, img):
        # 簡易白平衡處理
        b, g, r = cv2.split(img)
        b_avg, g_avg, r_avg = np.mean(b), np.mean(g), np.mean(r)
        
        if b_avg == 0 or g_avg == 0 or r_avg == 0:
            return img
            
        k = (b_avg + g_avg + r_avg) / 3
        kb, kg, kr = k / b_avg, k / g_avg, k / r_avg
        
        b = cv2.convertScaleAbs(b, alpha=kb)
        g = cv2.convertScaleAbs(g, alpha=kg)
        r = cv2.convertScaleAbs(r, alpha=kr)
        
        return cv2.merge([b, g, r])

    def draw_multiple_lane_lines(self, img, lines, thickness=6):
        height = img.shape[0]
        width = img.shape[1]  
        y_bottom = height               
        y_top = int(height * 0.6)       
        
        # 1. 變換車道邏輯
        if self.prev_left_lane is not None and self.prev_left_lane[0] > width * 0.5:
            self.prev_right_lane = self.prev_left_lane       
            self.prev_left_lane = self.prev_outer_left_lane  
            self.prev_outer_left_lane = None              
            self.left_miss_count = 0
            self.right_miss_count = 0

        elif self.prev_right_lane is not None and self.prev_right_lane[0] < width * 0.5:
            self.prev_left_lane = self.prev_right_lane       
            self.prev_right_lane = self.prev_outer_right_lane
            self.prev_outer_right_lane = None             
            self.left_miss_count = 0
            self.right_miss_count = 0

        def draw_lane(lane_data, color):
            if lane_data is not None:
                cv2.line(img, (lane_data[0], lane_data[1]), (lane_data[2], lane_data[3]), color, thickness)
                
        # 如果這幀完全沒抓到線
        if lines is None:
            self.left_miss_count += 1
            self.right_miss_count += 1
            self.outer_left_miss_count += 1
            self.outer_right_miss_count += 1
            
            if self.left_miss_count > self.MAX_MISS_FRAMES: self.prev_left_lane = None
            if self.right_miss_count > self.MAX_MISS_FRAMES: self.prev_right_lane = None
            if self.outer_left_miss_count > self.MAX_MISS_FRAMES: self.prev_outer_left_lane = None
            if self.outer_right_miss_count > self.MAX_MISS_FRAMES: self.prev_outer_right_lane = None
            
            draw_lane(self.prev_left_lane, [0, 0, 255])       
            draw_lane(self.prev_right_lane, [0, 0, 255])
            draw_lane(self.prev_outer_left_lane, [0, 255, 0]) 
            draw_lane(self.prev_outer_right_lane, [0, 255, 0])
            return img

        # 2. 底層線段過濾與分群
        y_anchor = int(height * 0.75) 
        lanes = [] 
        cluster_threshold = 100       

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 - x1 == 0: continue 
                slope = (y2 - y1) / (x2 - x1)
                
                if abs(slope) < 0.3: continue 
                
                x_horizon = (y_top - y1) / slope + x1
                if x_horizon < width * 0.05 or x_horizon > width * 0.95: continue
                
                x_bottom_ext = (y_bottom - y1) / slope + x1
                if slope < 0 and x_bottom_ext > width * 0.95: continue 
                if slope > 0 and x_bottom_ext < width * 0.05: continue 
                
                x_anchor = (y_anchor - y1) / slope + x1
                
                matched_lane = None
                for lane in lanes:
                    if abs(lane['x_anchor'] - x_anchor) < cluster_threshold:
                        matched_lane = lane
                        break
                
                if matched_lane:
                    matched_lane['points_x'].extend([x1, x2])
                    matched_lane['points_y'].extend([y1, y2])
                else:
                    lanes.append({'x_anchor': x_anchor, 'points_x': [x1, x2], 'points_y': [y1, y2]})

        # 3. 擬合多項式並分類左右候選線
        left_candidates = []
        right_candidates = []

        for lane in lanes:
            y_min = min(lane['points_y'])
            y_max = max(lane['points_y'])
            span = y_max - y_min 
            
            if span < height * 0.1: continue
                
            if len(lane['points_x']) >= 2: 
                fit = np.polyfit(lane['points_y'], lane['points_x'], 1)
                poly = np.poly1d(fit)
                
                x_start = int(poly(y_bottom))
                x_end = int(poly(y_top))
                
                if x_end - x_start == 0: continue
                lane_slope = (y_top - y_bottom) / (x_end - x_start)
                
                if lane_slope < 0:
                    left_candidates.append((x_start, y_bottom, x_end, y_top, span))
                else:
                    right_candidates.append((x_start, y_bottom, x_end, y_top, span))

        left_candidates.sort(key=lambda item: item[0], reverse=True)
        right_candidates.sort(key=lambda item: item[0])

        # 4. 記憶優先與擇優錄取邏輯
        current_left_lane = None
        if self.prev_left_lane is not None:
            best_cand = None
            min_dist = 80  
            for cand in left_candidates:
                dist_bottom = abs(cand[0] - self.prev_left_lane[0])
                dist_top = abs(cand[2] - self.prev_left_lane[2])
                if dist_bottom < min_dist and dist_top < 50:
                    min_dist = dist_bottom
                    best_cand = cand
            current_left_lane = best_cand
                    
        if current_left_lane is None:
            for cand in left_candidates:
                x_bottom, y_bottom, x_top, y_top, span = cand
                if x_bottom < width * 0.42 and (width * 0.35 < x_top < width * 0.65):
                    current_left_lane = cand
                    break

        outer_left_lane = None
        if current_left_lane is not None:
            curr_left_slope = (current_left_lane[3] - current_left_lane[1]) / (current_left_lane[2] - current_left_lane[0] + 1e-5)
            for cand in left_candidates:
                if cand == current_left_lane: continue
                cand_slope = (cand[3] - cand[1]) / (cand[2] - cand[0] + 1e-5)
                if cand[0] < current_left_lane[0] - 50 and cand[2] < current_left_lane[2] - 10:
                    if abs(cand_slope) < abs(curr_left_slope):
                        outer_left_lane = cand
                        break

        current_right_lane = None
        if self.prev_right_lane is not None:
            best_cand = None
            min_dist = 80  
            for cand in right_candidates:
                dist_bottom = abs(cand[0] - self.prev_right_lane[0])
                dist_top = abs(cand[2] - self.prev_right_lane[2])
                if dist_bottom < min_dist and dist_top < 50:
                    min_dist = dist_bottom
                    best_cand = cand
            current_right_lane = best_cand
                    
        if current_right_lane is None:
            for cand in right_candidates:
                x_bottom, y_bottom, x_top, y_top, span = cand
                if x_bottom > width * 0.58 and (width * 0.35 < x_top < width * 0.65):
                    current_right_lane = cand
                    break

        outer_right_lane = None
        if current_right_lane is not None:
            curr_right_slope = (current_right_lane[3] - current_right_lane[1]) / (current_right_lane[2] - current_right_lane[0] + 1e-5)
            for cand in right_candidates:
                if cand == current_right_lane: continue
                cand_slope = (cand[3] - cand[1]) / (cand[2] - cand[0] + 1e-5)
                if cand[0] > current_right_lane[0] + 50 and cand[2] > current_right_lane[2] + 10:
                    if abs(cand_slope) < abs(curr_right_slope):
                        outer_right_lane = cand
                        break
                    
        # 5. 頂部寬度與斜率對稱性防護網
        if current_left_lane is not None and current_right_lane is not None:
            top_gap = current_right_lane[2] - current_left_lane[2]
            left_slope = (current_left_lane[3] - current_left_lane[1]) / (current_left_lane[2] - current_left_lane[0] + 1e-5)
            right_slope = (current_right_lane[3] - current_right_lane[1]) / (current_right_lane[2] - current_right_lane[0] + 1e-5)
            
            if top_gap < width * 0.05 or abs(abs(left_slope) - abs(right_slope)) > 0.8:
                current_left_lane = None
                current_right_lane = None

        # 6. 平滑處理與防跳躍
        def smooth_and_prevent_jumping(curr_lane, prev_lane, miss_count, is_left, base_max_shift=30, alpha=0.1):
            if curr_lane is None: return prev_lane, miss_count + 1
            if prev_lane is None: return curr_lane, 0
                
            curr_x_bottom, curr_x_top = curr_lane[0], curr_lane[2]
            prev_x_bottom, prev_x_top = prev_lane[0], prev_lane[2]
            
            current_max_shift = min(base_max_shift + (miss_count * 2), base_max_shift + 30)
            shift_bottom = curr_x_bottom - prev_x_bottom
            shift_top = curr_x_top - prev_x_top
            
            if is_left and shift_bottom > current_max_shift:
                if shift_top > -current_max_shift: return curr_lane, 0 
            if not is_left and shift_bottom < -current_max_shift:
                if shift_top < current_max_shift: return curr_lane, 0 
                
            if abs(shift_bottom) > current_max_shift or abs(shift_top) > current_max_shift:
                return prev_lane, miss_count + 1 
                
            smooth_x_bottom = int(prev_x_bottom * (1 - alpha) + curr_x_bottom * alpha)
            smooth_x_top = int(prev_x_top * (1 - alpha) + curr_x_top * alpha)
            
            return (smooth_x_bottom, curr_lane[1], smooth_x_top, curr_lane[3], curr_lane[4]), 0

        current_left_lane, self.left_miss_count = smooth_and_prevent_jumping(
            current_left_lane, self.prev_left_lane, self.left_miss_count, is_left=True)
        current_right_lane, self.right_miss_count = smooth_and_prevent_jumping(
            current_right_lane, self.prev_right_lane, self.right_miss_count, is_left=False)
            
        outer_left_lane, self.outer_left_miss_count = smooth_and_prevent_jumping(
            outer_left_lane, self.prev_outer_left_lane, self.outer_left_miss_count, is_left=True, base_max_shift=40)
        outer_right_lane, self.outer_right_miss_count = smooth_and_prevent_jumping(
            outer_right_lane, self.prev_outer_right_lane, self.outer_right_miss_count, is_left=False, base_max_shift=40)

        # 7. 最終邏輯檢查與繪圖
        if current_left_lane and outer_left_lane:
            if outer_left_lane[0] >= current_left_lane[0] or outer_left_lane[2] >= current_left_lane[2]:
                outer_left_lane = None
                self.outer_left_miss_count = self.MAX_MISS_FRAMES + 1

        if current_right_lane and outer_right_lane:
            if outer_right_lane[0] <= current_right_lane[0] or outer_right_lane[2] <= current_right_lane[2]:
                outer_right_lane = None
                self.outer_right_miss_count = self.MAX_MISS_FRAMES + 1

        if self.left_miss_count > self.MAX_MISS_FRAMES: current_left_lane = None
        if self.right_miss_count > self.MAX_MISS_FRAMES: current_right_lane = None
        if self.outer_left_miss_count > self.MAX_MISS_FRAMES: outer_left_lane = None
        if self.outer_right_miss_count > self.MAX_MISS_FRAMES: outer_right_lane = None

        self.prev_left_lane = current_left_lane
        self.prev_right_lane = current_right_lane
        self.prev_outer_left_lane = outer_left_lane
        self.prev_outer_right_lane = outer_right_lane

        draw_lane(self.prev_left_lane, [0, 0, 255])       
        draw_lane(self.prev_right_lane, [0, 0, 255])      
        draw_lane(self.prev_outer_left_lane, [0, 255, 0]) 
        draw_lane(self.prev_outer_right_lane, [0, 255, 0])
                
        return img

    def process_frame(self, frame):
        height, width = frame.shape[:2]
        balanced_frame = self.apply_white_balance(frame)

        hls = cv2.cvtColor(balanced_frame, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls)
        
        avg_brightness = np.mean(l)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        hls_eq = cv2.merge([h, l_eq, s])

        if avg_brightness < 80:
            white_l_lower = 110  
            yellow_s_lower = 70  
        else:
            white_l_lower = 180  
            yellow_s_lower = 100 

        lower_white = np.array([0, white_l_lower, 0]) 
        upper_white = np.array([255, 255, 255])
        mask_white = cv2.inRange(hls_eq, lower_white, upper_white)
        
        lower_yellow = np.array([10, 0, yellow_s_lower])
        upper_yellow = np.array([40, 255, 255])
        mask_yellow = cv2.inRange(hls_eq, lower_yellow, upper_yellow)
        
        mask_color = cv2.bitwise_or(mask_white, mask_yellow)
        color_filtered = cv2.bitwise_and(balanced_frame, balanced_frame, mask=mask_color)

        gray = cv2.cvtColor(color_filtered, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        polygons = np.array([[
            (-100, height),                      
            (width + 100, height),               
            (int(width * 0.9), int(height * 0.55)), 
            (int(width * 0.1), int(height * 0.55))  
        ]], dtype=np.int32)
        
        mask_roi = np.zeros_like(edges)
        cv2.fillPoly(mask_roi, polygons, 255)
        masked_edges = cv2.bitwise_and(edges, mask_roi)
        
        binary_bgr = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
        
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=200)
        
        line_image = np.zeros_like(frame)
        self.draw_multiple_lane_lines(line_image, lines)
        
        left_frame = cv2.addWeighted(balanced_frame, 0.8, line_image, 1.0, 0)
        right_frame = cv2.addWeighted(binary_bgr, 0.8, line_image, 1.0, 0)
        
        return np.hstack((left_frame, right_frame))
