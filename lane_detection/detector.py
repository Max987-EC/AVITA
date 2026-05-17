import cv2
import numpy as np
from .processor import LaneImageProcessor
from .tracker import LaneTracker

class LaneDetector:
    def __init__(self):
        self.tracker = LaneTracker()

    def _find_lane_candidates(self, lines, width, height):
        if lines is None: return None, None, None, None
            
        y_bottom, y_top = height, int(height * 0.6)
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
                
                matched_lane = next((l for l in lanes if abs(l['x_anchor'] - x_anchor) < cluster_threshold), None)
                if matched_lane:
                    matched_lane['points_x'].extend([x1, x2])
                    matched_lane['points_y'].extend([y1, y2])
                else:
                    lanes.append({'x_anchor': x_anchor, 'points_x': [x1, x2], 'points_y': [y1, y2]})

        left_cands, right_cands = [], []
        for lane in lanes:
            span = max(lane['points_y']) - min(lane['points_y'])
            if span < height * 0.1 or len(lane['points_x']) < 2: continue
                
            poly = np.poly1d(np.polyfit(lane['points_y'], lane['points_x'], 1))
            x_start, x_end = int(poly(y_bottom)), int(poly(y_top))
            
            if x_end - x_start == 0: continue
            if ((y_top - y_bottom) / (x_end - x_start)) < 0:
                left_cands.append((x_start, y_bottom, x_end, y_top, span))
            else:
                right_cands.append((x_start, y_bottom, x_end, y_top, span))

        left_cands.sort(key=lambda item: item[0], reverse=True)
        right_cands.sort(key=lambda item: item[0])

        def get_best_lane(cands, prev_lane, is_left):
            if prev_lane:
                valid = [c for c in cands if abs(c[0] - prev_lane[0]) < 80 and abs(c[2] - prev_lane[2]) < 50]
                if valid: return min(valid, key=lambda c: abs(c[0] - prev_lane[0]))
            
            for c in cands:
                if (is_left and c[0] < width * 0.42) or (not is_left and c[0] > width * 0.58):
                    if width * 0.35 < c[2] < width * 0.65: return c
            return None

        curr_left = get_best_lane(left_cands, self.tracker.prev_left, True)
        curr_right = get_best_lane(right_cands, self.tracker.prev_right, False)

        def get_outer_lane(cands, curr_lane, is_left):
            if not curr_lane: return None
            curr_slope = (curr_lane[3] - curr_lane[1]) / (curr_lane[2] - curr_lane[0] + 1e-5)
            for c in cands:
                if c == curr_lane: continue
                c_slope = (c[3] - c[1]) / (c[2] - c[0] + 1e-5)
                if (is_left and c[0] < curr_lane[0] - 50) or (not is_left and c[0] > curr_lane[0] + 50):
                    if abs(c_slope) < abs(curr_slope): return c
            return None

        outer_left = get_outer_lane(left_cands, curr_left, True)
        outer_right = get_outer_lane(right_cands, curr_right, False)

        if curr_left and curr_right:
            l_slope = (curr_left[3] - curr_left[1]) / (curr_left[2] - curr_left[0] + 1e-5)
            r_slope = (curr_right[3] - curr_right[1]) / (curr_right[2] - curr_right[0] + 1e-5)
            if (curr_right[2] - curr_left[2]) < width * 0.05 or abs(abs(l_slope) - abs(r_slope)) > 0.8:
                curr_left, curr_right = None, None

        return curr_left, curr_right, outer_left, outer_right

    def process_frame(self, frame):
        height, width = frame.shape[:2]
        self.tracker.check_lane_change(width)

        balanced_frame, masked_edges = LaneImageProcessor.get_masked_edges(frame)
        binary_bgr = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
        
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=200)
        
        c_left, c_right, o_left, o_right = self._find_lane_candidates(lines, width, height)
        f_left, f_right, f_o_left, f_o_right = self.tracker.update(c_left, c_right, o_left, o_right)

        line_image = np.zeros_like(frame)
        def draw(lane, color):
            if lane: cv2.line(line_image, (lane[0], lane[1]), (lane[2], lane[3]), color, 6)
                
        draw(f_left, [0, 0, 255])
        draw(f_right, [0, 0, 255])
        draw(f_o_left, [0, 255, 0])
        draw(f_o_right, [0, 255, 0])
        
        left_frame = cv2.addWeighted(balanced_frame, 0.8, line_image, 1.0, 0)
        right_frame = cv2.addWeighted(binary_bgr, 0.8, line_image, 1.0, 0)
        
        return np.hstack((left_frame, right_frame))
