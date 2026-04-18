import cv2
import numpy as np
import base64 # 🌟 新增：用於直方圖的 Base64 編碼

class ImageProcessor:
    def __init__(self, img_array):
        # 網頁上傳的圖片通常是彩色 (BGR)，我們在這裡統一轉為灰階處理
        if len(img_array.shape) == 3:
            self.img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            self.img = img_array

    # ==========================================
    # 🌟 新增：直方圖生成 (封裝在類別內)
    # ==========================================
    @staticmethod
    def generate_histogram_base64(img):
        h, w = 300, 400
        # 建立深色背景畫布，對應前端的 #0a0f19
        hist_img = np.zeros((h, w, 3), dtype=np.uint8)
        hist_img[:] = (25, 15, 10) # BGR 格式

        # 判斷是灰階還是彩色影像
        if len(img.shape) == 2 or img.shape[2] == 1:
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, h, cv2.NORM_MINMAX)
            for x in range(256):
                cv2.line(hist_img, (int(x * w / 256), h), (int(x * w / 256), h - int(hist[x])), (200, 200, 200), 1)
        else:
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] # B, G, R
            for i, col in enumerate(colors):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                cv2.normalize(hist, hist, 0, h, cv2.NORM_MINMAX)
                for x in range(1, 256):
                    cv2.line(hist_img, 
                             (int((x - 1) * w / 256), h - int(hist[x - 1])),
                             (int(x * w / 256), h - int(hist[x])), 
                             col, 2)
        
        # 將直方圖轉為 Base64
        _, buffer = cv2.imencode('.png', hist_img)
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
