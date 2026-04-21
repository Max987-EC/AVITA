import cv2
import numpy as np
import base64

class ImageProcessor:
    def __init__(self, img_array):
        if len(img_array.shape) == 3:
            self.img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            self.img = img_array

    # ==========================================
    # 🌟 升級：直方圖生成 (主副刻度高密度版)
    # ==========================================
    @staticmethod
    def generate_histogram_base64(img):
        h, w = 300, 400
        pad_bottom = 30 # 預留底部空間畫座標軸
        
        hist_img = np.zeros((h + pad_bottom, w, 3), dtype=np.uint8)
        hist_img[:] = (25, 15, 10) 

        # 畫一條 X 軸底線
        cv2.line(hist_img, (0, h), (w, h), (100, 100, 100), 1)

        # 繪製直方圖曲線
        if len(img.shape) == 2 or img.shape[2] == 1:
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, h, cv2.NORM_MINMAX)
            for x in range(256):
                # 🌟 這裡的 hist[x] 改成 hist[x][0]
                cv2.line(hist_img, (int(x * w / 256), h), (int(x * w / 256), h - int(hist[x][0])), (200, 200, 200), 1)
        else:
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] 
            for i, col in enumerate(colors):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                cv2.normalize(hist, hist, 0, h, cv2.NORM_MINMAX)
                for x in range(1, 256):
                    # 🌟 這裡的 hist[x-1] 與 hist[x] 都加上 [0]
                    cv2.line(hist_img, 
                             (int((x - 1) * w / 256), h - int(hist[x - 1][0])),
                             (int(x * w / 256), h - int(hist[x][0])), 
                             col, 2)
        
        # 🌟 繪製高密度 X 軸刻度與文字 (主副刻度設計)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        
        # 每隔 16 畫一個刻度，總共會有 17 個刻度點
        for val in range(0, 257, 16):
            # 處理最後一個邊界值 255
            if val == 256:
                val = 255
                
            x_pos = int(val * (w - 1) / 255)
            
            # 每隔 32 畫「主刻度」(長線 + 數字)，其餘畫「副刻度」(短線)
            if val % 32 == 0 or val == 255:
                # 主刻度：線條較長 (向下 8px)、顏色較亮
                cv2.line(hist_img, (x_pos, h), (x_pos, h + 8), (180, 180, 180), 1)
                
                text = str(val)
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = x_pos - (text_width // 2)
                
                # 邊界防呆：確保左右邊緣的數字不會被裁切
                if text_x < 0:
                    text_x = 2
                elif text_x + text_width > w:
                    text_x = w - text_width - 2
                    
                cv2.putText(hist_img, text, (text_x, h + 22), font, font_scale, (180, 180, 180), thickness, cv2.LINE_AA)
            else:
                # 副刻度：線條較短 (向下 4px)、顏色較暗，且不標示文字以免擁擠
                cv2.line(hist_img, (x_pos, h), (x_pos, h + 4), (100, 100, 100), 1)

        _, buffer = cv2.imencode('.png', hist_img)
        return base64.b64encode(buffer).decode('utf-8')

    # ==========================================
    # 🌟 升級：頻譜圖生成 (純灰階專業版)
    # ==========================================
    @staticmethod
    def generate_spectrum_base64(img):
        # 1. 確保影像是單通道灰階，以確保傅立葉轉換的準確性
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 2. 執行二維快速傅立葉轉換 (FFT)
        f = np.fft.fft2(gray)
        # 將零頻率 (DC 分量) 移到頻譜中心
        fshift = np.fft.fftshift(f)
        
        # 3. 計算強度頻譜 (取對數以壓縮極大的動態範圍，讓細節可見)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # 4. 正規化到 0~255 的 8-bit 灰階影像
        spectrum_img = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 💡 移除了偽色彩 (applyColorMap) 的渲染，直接輸出純灰階影像
        _, buffer = cv2.imencode('.jpg', spectrum_img)
        return base64.b64encode(buffer).decode('utf-8')

    # ==========================================
    # 1. 強度轉換函數
    # ==========================================
    def binarize(self, threshold=127):
        _, result = cv2.threshold(self.img, threshold, 255, cv2.THRESH_BINARY)
        return result

    def negative_transform(self):
        return 255 - self.img

    def log_transform(self, c=1.0):
        img_float = np.float32(self.img)
        result = c * np.log(1 + img_float)
        return np.uint8(cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX))

    def power_law_transform(self, gamma=1.0, c=1.0):
        img_float = np.float32(self.img) / 255.0
        result = c * np.power(img_float, gamma)
        return np.uint8(np.clip(result * 255.0, 0, 255))

    def histogram_equalization(self):
        return cv2.equalizeHist(self.img)

    # ==========================================
    # 2. 影像空間濾波
    # ==========================================
    def mean_filter(self, kernel_size=3):
        return cv2.blur(self.img, (kernel_size, kernel_size))

    def gaussian_filter(self, kernel_size=3, sigma=1.0):
        return cv2.GaussianBlur(self.img, (kernel_size, kernel_size), sigma)

    def median_filter(self, kernel_size=3):
        return cv2.medianBlur(self.img, kernel_size)

    def sobel_filter(self):
        sobel_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobel_x, sobel_y)
        return np.uint8(cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX))

    def laplacian_filter(self):
        laplacian = cv2.Laplacian(self.img, cv2.CV_64F)
        return cv2.convertScaleAbs(laplacian)

    # ==========================================
    # 3. 影像頻率濾波
    # ==========================================
    def _get_frequency_components(self):
        f_transform = np.fft.fft2(self.img)
        return np.fft.fftshift(f_transform)

    def _apply_frequency_filter(self, mask):
        f_shift = self._get_frequency_components()
        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.abs(np.fft.ifft2(f_ishift))
        return np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))

    def _create_distance_matrix(self):
        rows, cols = self.img.shape
        crow, ccol = rows // 2, cols // 2
        u = np.arange(rows)
        v = np.arange(cols)
        V, U = np.meshgrid(v, u)
        return np.sqrt((U - crow)**2 + (V - ccol)**2)

    def frequency_filter(self, filter_type='gaussian', pass_type='low', D0=30, n=2):
        D = self._create_distance_matrix()
        mask = np.zeros_like(D)

        if filter_type == 'ideal':
            if pass_type == 'low': mask[D <= D0] = 1
            else: mask[D > D0] = 1
        elif filter_type == 'butterworth':
            D_safe = np.where(D == 0, 1e-5, D)
            if pass_type == 'low': mask = 1 / (1 + (D_safe / D0)**(2 * n))
            else: mask = 1 / (1 + (D0 / D_safe)**(2 * n))
        elif filter_type == 'gaussian':
            if pass_type == 'low': mask = np.exp(-(D**2) / (2 * (D0**2)))
            else: mask = 1 - np.exp(-(D**2) / (2 * (D0**2)))

        return self._apply_frequency_filter(mask)
