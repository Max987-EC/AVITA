import cv2
import numpy as np

class LaneImageProcessor:
    @staticmethod
    def apply_white_balance(img):
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

    @staticmethod
    def get_masked_edges(frame):
        height, width = frame.shape[:2]
        balanced_frame = LaneImageProcessor.apply_white_balance(frame)

        hls = cv2.cvtColor(balanced_frame, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls)
        
        avg_brightness = np.mean(l)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        hls_eq = cv2.merge([h, l_eq, s])

        white_l_lower = 110 if avg_brightness < 80 else 180
        yellow_s_lower = 70 if avg_brightness < 80 else 100

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
        
        return balanced_frame, masked_edges
