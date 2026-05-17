class LaneTracker:
    def __init__(self, max_miss_frames=60):
        self.MAX_MISS_FRAMES = max_miss_frames
        
        self.prev_left = None
        self.prev_right = None
        self.left_miss = 0
        self.right_miss = 0

        self.prev_outer_left = None
        self.prev_outer_right = None
        self.outer_left_miss = 0
        self.outer_right_miss = 0

    def check_lane_change(self, width):
        if self.prev_left is not None and self.prev_left[0] > width * 0.5:
            self.prev_right = self.prev_left       
            self.prev_left = self.prev_outer_left  
            self.prev_outer_left = None              
            self.left_miss = 0
            self.right_miss = 0

        elif self.prev_right is not None and self.prev_right[0] < width * 0.5:
            self.prev_left = self.prev_right       
            self.prev_right = self.prev_outer_right
            self.prev_outer_right = None             
            self.left_miss = 0
            self.right_miss = 0

    def _smooth_lane(self, curr_lane, prev_lane, miss_count, is_left, base_max_shift=30, alpha=0.1):
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

    def update(self, curr_left, curr_right, outer_left, outer_right):
        curr_left, self.left_miss = self._smooth_lane(curr_left, self.prev_left, self.left_miss, True)
        curr_right, self.right_miss = self._smooth_lane(curr_right, self.prev_right, self.right_miss, False)
        outer_left, self.outer_left_miss = self._smooth_lane(outer_left, self.prev_outer_left, self.outer_left_miss, True, 40)
        outer_right, self.outer_right_miss = self._smooth_lane(outer_right, self.prev_outer_right, self.outer_right_miss, False, 40)

        if curr_left and outer_left and (outer_left[0] >= curr_left[0] or outer_left[2] >= curr_left[2]):
            outer_left = None
            self.outer_left_miss = self.MAX_MISS_FRAMES + 1

        if curr_right and outer_right and (outer_right[0] <= curr_right[0] or outer_right[2] <= curr_right[2]):
            outer_right = None
            self.outer_right_miss = self.MAX_MISS_FRAMES + 1

        if self.left_miss > self.MAX_MISS_FRAMES: curr_left = None
        if self.right_miss > self.MAX_MISS_FRAMES: curr_right = None
        if self.outer_left_miss > self.MAX_MISS_FRAMES: outer_left = None
        if self.outer_right_miss > self.MAX_MISS_FRAMES: outer_right = None

        self.prev_left, self.prev_right = curr_left, curr_right
        self.prev_outer_left, self.prev_outer_right = outer_left, outer_right

        return self.prev_left, self.prev_right, self.prev_outer_left, self.prev_outer_right
