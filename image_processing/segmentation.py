# 影像分割模組 (Image Segmentation Module)
# 包含將影像劃分為具備相似屬性區域的各種演算法。

import cv2
import numpy as np

class SegmentationMixin:
    """
    影像分割混入類別 (Image Segmentation Mixin):
    提供區域成長 (漫水填充) 與區域分裂合併 (四分樹) 等技術。
    """
    
    def region_growing(self, seed_x=None, seed_y=None, tolerance=20):
        """
        區域成長 / 漫水填充 (Region Growing / Flood Fill):
        從種子點開始，將相鄰且灰階值差異在容差範圍內的像素合併為同一區域。
        """
        h, w = self.img.shape[:2]
        
        # 預設種子點設為影像中心
        if seed_x is None or seed_y is None:
            seed_x, seed_y = w // 2, h // 2
        else:
            # 座標邊界限制，防止越界
            seed_x = min(max(int(seed_x), 0), w - 1)
            seed_y = min(max(int(seed_y), 0), h - 1)
            
        seed_point = (seed_x, seed_y)
        
        # OpenCV floodFill 需要一個比原圖四周各寬 1 像素的遮罩 (Mask)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        output_img = self.img.copy()
        
        # 設定填滿顏色與容差範圍 (lo_diff 為低方向容差, up_diff 為高方向容差)
        lo_diff = (tolerance, tolerance, tolerance) if len(self.img.shape) == 3 else (tolerance,)
        up_diff = (tolerance, tolerance, tolerance) if len(self.img.shape) == 3 else (tolerance,)
        
        # 執行填充：FLOODFILL_FIXED_RANGE 代表以種子點像素值作為基準進行比較
        cv2.floodFill(output_img, mask, seed_point, (0, 255, 0), lo_diff, up_diff, cv2.FLOODFILL_FIXED_RANGE)
        
        self.steps.append((f"Step 1: 區域成長結果 (種子: {seed_x},{seed_y}, 容差: {tolerance})", output_img))
        return output_img

    def region_split_merge(self, std_threshold=15, min_size=8):
        """
        區域分裂與合併 (Region Split and Merge):
        利用四分樹 (Quadtree) 結構，將影像遞迴分裂為均勻區域，若區域不均勻則繼續分裂。
        """
        gray = self._get_gray()
        h, w = gray.shape
        output = np.zeros_like(gray)
        
        # 遞迴核心函數
        def split(x, y, w_box, h_box):
            # 停止條件 1: 區域已小於最小尺寸限制
            if w_box <= min_size or h_box <= min_size:
                output[y:y+h_box, x:x+w_box] = np.mean(gray[y:y+h_box, x:x+w_box])
                return
            
            # 計算該區域的標準差 (反映區域的均勻度)
            roi = gray[y:y+h_box, x:x+w_box]
            _, std_val = cv2.meanStdDev(roi)
            
            # 判斷是否需要進一步分裂
            if std_val[0][0] > std_threshold:
                # 區域不均勻 -> 分裂為四個象限
                w2, h2 = w_box // 2, h_box // 2
                split(x, y, w2, h2)                             # 左上 (Top-Left)
                split(x + w2, y, w_box - w2, h2)                # 右上 (Top-Right)
                split(x, y + h2, w2, h_box - h2)                # 左下 (Bottom-Left)
                split(x + w2, y + h2, w_box - w2, h_box - h2)   # 右下 (Bottom-Right)
            else:
                # 區域均勻 -> 停止分裂並填入平均值 (隱含的合併動作)
                mean_val, _ = cv2.meanStdDev(roi)
                output[y:y+h_box, x:x+w_box] = mean_val[0][0]
                
        # 從整張影像啟動遞迴
        split(0, 0, w, h)
        
        self.steps.append((f"Step 1: 區域分裂合併結果 (門檻: {std_threshold}, 最小尺寸: {min_size})", output))
        return output
