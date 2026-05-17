import cv2
import numpy as np

class SegmentationMixin:
    
    def region_growing(self, seed_x=None, seed_y=None, tolerance=20):
        """區域成長 (漫水填充)"""
        h, w = self.img.shape[:2]
        
        # 如果沒有指定種子點，預設取影像正中央
        if seed_x is None or seed_y is None:
            seed_x = w // 2
            seed_y = h // 2
        else:
            # [新增] 限制座標範圍，防止越界崩潰
            seed_x = min(max(int(seed_x), 0), w - 1)
            seed_y = min(max(int(seed_y), 0), h - 1)
            
        seed_point = (seed_x, seed_y)
        
        # 漫水填充需要一個比原圖大 2 個像素的遮罩
        mask = np.zeros((h + 2, w + 2), np.uint8)
        
        output_img = self.img.copy()
        
        # 執行 FloodFill
        lo_diff = (tolerance, tolerance, tolerance) if len(self.img.shape) == 3 else (tolerance,)
        up_diff = (tolerance, tolerance, tolerance) if len(self.img.shape) == 3 else (tolerance,)
        
        cv2.floodFill(output_img, mask, seed_point, (0, 255, 0), lo_diff, up_diff, cv2.FLOODFILL_FIXED_RANGE)
        
        self.steps.append((f"Step 1: 區域成長 (種子點: {seed_x},{seed_y}, 容差: {tolerance})", output_img))
        return output_img

    def region_split_merge(self, std_threshold=15, min_size=8):
        """區域分裂與合併 (Quadtree Split and Merge)"""
        gray = self._get_gray()
        h, w = gray.shape
        output = np.zeros_like(gray)
        
        # 遞迴函數：執行四分樹分裂
        def split(x, y, w_box, h_box):
            # 如果區塊已經小於最小尺寸，直接填入平均顏色並停止分裂
            if w_box <= min_size or h_box <= min_size:
                output[y:y+h_box, x:x+w_box] = np.mean(gray[y:y+h_box, x:x+w_box])
                return
            
            # 計算該區塊的標準差 (判斷紋理是否複雜)
            roi = gray[y:y+h_box, x:x+w_box]
            mean_val, std_val = cv2.meanStdDev(roi)
            
            # 如果標準差大於門檻，代表區域不均勻，繼續切成 4 份
            if std_val[0][0] > std_threshold:
                w2, h2 = w_box // 2, h_box // 2
                split(x, y, w2, h2)                             # 左上
                split(x + w2, y, w_box - w2, h2)                # 右上
                split(x, y + h2, w2, h_box - h2)                # 左下
                split(x + w2, y + h2, w_box - w2, h_box - h2)   # 右下
            else:
                # 區域夠均勻，直接填滿平均顏色 (這就是合併的雛形)
                output[y:y+h_box, x:x+w_box] = mean_val[0][0]
                
        # 從整張圖片開始分裂
        split(0, 0, w, h)
        
        # [修正] 補上 Step 1 確保 UI 一致性
        self.steps.append((f"Step 1: 區域分裂與合併 (Std TH: {std_threshold}, Min Size: {min_size})", output))
        return output
