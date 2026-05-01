import cv2
import numpy as np
import base64

class ImageProcessor:
    """
    綜合影像處理核心類別
    負責處理所有與影像相關的演算法，包含：強度轉換、空間濾波、邊緣偵測與頻率域濾波。
    """
    def __init__(self, img_array):
        # 儲存原始影像陣列
        self.img = img_array
        # 判斷影像是否為彩色 (若維度長度為 3，代表具有 BGR 三個通道)
        self.is_color = len(img_array.shape) == 3
        # 🌟 新增：用來儲存執行步驟的列表 [(步驟名稱, 影像陣列), ...]
        self.steps = [] 

    # 🌟 新增：紀錄步驟的輔助函式
    def add_step(self, name, img):
        """將當前的影像狀態加入步驟紀錄中"""
        self.steps.append((name, img.copy()))

    # ==========================================
    # 📊 基礎工具：直方圖與頻譜圖生成
    # ==========================================
    @staticmethod
    def generate_histogram_base64(img):
        """
        繪製影像的直方圖，並回傳 Base64 編碼的圖片字串。
        支援灰階與彩色影像，並帶有精緻的座標軸與刻度標示。
        """
        h, w = 300, 400
        pad_bottom = 30 # 預留底部空間繪製 X 軸與刻度文字
        
        # 建立深色背景的畫布
        hist_img = np.zeros((h + pad_bottom, w, 3), dtype=np.uint8)
        hist_img[:] = (25, 15, 10) 

        # 繪製 X 軸底線
        cv2.line(hist_img, (0, h), (w, h), (100, 100, 100), 1)

        # 🌟 判斷影像本質是否為灰階 (加大智能容錯機制)
        is_gray = False
        if len(img.shape) == 2 or img.shape[2] == 1:
            is_gray = True
        elif len(img.shape) == 3:
            b, g, r = cv2.split(img)
            diff_bg = cv2.absdiff(b, g)
            diff_gr = cv2.absdiff(g, r)
            color_mask = cv2.bitwise_or(diff_bg, diff_gr)
            color_pixels = cv2.countNonZero(color_mask)
            total_pixels = img.shape[0] * img.shape[1]
            
            # 💡 終極容錯機制：只要彩色像素佔比不到 50%，我們就視為灰階圖
            # 這樣即使畫了滿滿的霍夫圓圈或線條，依然能保持灰階直方圖
            if color_pixels / total_pixels < 0.5:
                is_gray = True
                # 將帶有彩色線條的圖轉回純灰階，以便計算直方圖
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if is_gray:
            # 灰階直方圖：計算並正規化後畫出「灰色」折線
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, h, cv2.NORM_MINMAX)
            for x in range(1, 256):
                cv2.line(hist_img, 
                         (int((x - 1) * w / 256), h - int(hist[x - 1][0])),
                         (int(x * w / 256), h - int(hist[x][0])), 
                         (150, 150, 150), 2) # 使用灰色
        else:
            # 彩色直方圖：分別計算 B, G, R 三個通道並畫出對應顏色的折線
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] # OpenCV 預設為 BGR 順序
            for i, col in enumerate(colors):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                cv2.normalize(hist, hist, 0, h, cv2.NORM_MINMAX)
                for x in range(1, 256):
                    cv2.line(hist_img, 
                             (int((x - 1) * w / 256), h - int(hist[x - 1][0])),
                             (int(x * w / 256), h - int(hist[x][0])), 
                             col, 2)
        
        # 繪製 X 軸的刻度與數值標籤 (每隔 16 畫一個刻度)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        
        for val in range(0, 257, 16):
            if val == 256: val = 255
            x_pos = int(val * (w - 1) / 255)
            
            # 每隔 32 畫主刻度並標示數字，其餘畫副刻度
            if val % 32 == 0 or val == 255:
                cv2.line(hist_img, (x_pos, h), (x_pos, h + 8), (180, 180, 180), 1)
                text = str(val)
                (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                # 防呆機制：確保邊緣的數字不會超出畫布被裁切
                text_x = x_pos - (text_width // 2)
                if text_x < 0: text_x = 2
                elif text_x + text_width > w: text_x = w - text_width - 2
                
                cv2.putText(hist_img, text, (text_x, h + 22), font, font_scale, (180, 180, 180), thickness, cv2.LINE_AA)
            else:
                cv2.line(hist_img, (x_pos, h), (x_pos, h + 4), (100, 100, 100), 1)

        # 編碼為 PNG 格式的 Base64 字串
        _, buffer = cv2.imencode('.png', hist_img)
        return base64.b64encode(buffer).decode('utf-8')

    @staticmethod
    def generate_spectrum_base64(img):
        """
        計算影像的二維傅立葉轉換，並回傳視覺化頻譜圖的 Base64 字串。
        """
        # 頻譜分析通常在灰階下進行以觀察整體結構
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 執行 FFT 並將低頻(DC分量)移至中心
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        # 取對數以壓縮極大的動態範圍，讓頻譜細節可見
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # 正規化到 0~255 以便顯示
        spectrum_img = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, buffer = cv2.imencode('.jpg', spectrum_img)
        return base64.b64encode(buffer).decode('utf-8')

    # ==========================================
    # ⚙️ 輔助函式
    # ==========================================
    def _get_gray(self):
        """
        安全獲取灰階影像。
        許多邊緣偵測演算法(如 Canny, Sobel)僅支援或適合在單通道下運作。
        """
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) if self.is_color else self.img

    # ==========================================
    # 1️⃣ 基本強度轉換 (Intensity Transformations)
    # ==========================================
    def binarize(self, threshold=127):
        """二值化：將影像轉為純黑與純白"""
        gray = self._get_gray()
        _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return result

    def negative_transform(self):
        """負轉換：像素值反轉 (255 - pixel)"""
        return 255 - self.img

    def log_transform(self, c=1.0):
        """對數轉換：擴展暗部細節，壓縮亮部細節"""
        img_float = np.float32(self.img)
        result = c * np.log(1 + img_float)
        return np.uint8(cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX))

    def power_law_transform(self, gamma=1.0, c=1.0):
        """指數(Gamma)轉換：gamma < 1 提亮暗部，gamma > 1 壓暗亮部"""
        img_float = np.float32(self.img) / 255.0
        result = c * np.power(img_float, gamma)
        return np.uint8(np.clip(result * 255.0, 0, 255))

    def histogram_equalization(self):
        """直方圖等化：全域對比度增強"""
        if self.is_color:
            # 彩色影像：轉為 HSV，僅對 V (亮度) 通道進行等化，避免色偏
            hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return cv2.equalizeHist(self.img)

    def clahe_equalization(self, clip_limit=2.0, tile_grid_size=8):
        """CLAHE：限制對比自適應直方圖等化 (局部對比度增強，抑制雜訊)"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        if self.is_color:
            # 彩色影像：同樣在 HSV 空間下僅處理 V 通道
            hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return clahe.apply(self.img)

    # ==========================================
    # 2️⃣ 影像空間濾波與邊緣偵測 (Spatial Filtering)
    # ==========================================
    def mean_filter(self, kernel_size=3):
        """均值濾波：平滑影像，降低雜訊 (會使邊緣模糊)"""
        return cv2.blur(self.img, (kernel_size, kernel_size))

    def gaussian_filter(self, kernel_size=3, sigma=1.0):
        """高斯濾波：保留較多邊緣細節的平滑化處理"""
        return cv2.GaussianBlur(self.img, (kernel_size, kernel_size), sigma)

    def median_filter(self, kernel_size=3):
        """中值濾波：有效去除胡椒鹽雜訊"""
        return cv2.medianBlur(self.img, kernel_size)

    def roberts_filter(self):
        """Roberts 算子：利用 2x2 交叉微分偵測邊緣 (對雜訊敏感)"""
        gray = self._get_gray()
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
        magnitude = cv2.magnitude(x, y)
        return np.uint8(cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX))

    def prewitt_filter(self):
        """Prewitt 算子：利用 3x3 平均微分偵測邊緣"""
        gray = self._get_gray()
        kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
        magnitude = cv2.magnitude(x, y)
        return np.uint8(cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX))

    def sobel_filter(self):
        """Sobel 算子：結合高斯平滑與微分，工業界最常用的邊緣偵測"""
        gray = self._get_gray()
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobel_x, sobel_y)
        return np.uint8(cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX))

    def laplacian_filter(self):
        """Laplacian 算子：二階微分邊緣偵測 (可抓出更細的邊緣，但極怕雜訊)"""
        gray = self._get_gray()
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return cv2.convertScaleAbs(laplacian)

    def log_filter(self, kernel_size=3, sigma=1.0):
        """LoG (Laplacian of Gaussian)：先高斯模糊降噪，再做拉普拉斯邊緣偵測"""
        gray = self._get_gray()
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        return cv2.convertScaleAbs(laplacian)

    def canny_filter(self, threshold1=50, threshold2=150):
        """Canny 算子：五步驟最佳化邊緣偵測，能輸出連續且細緻的邊緣 (加入步驟紀錄)"""
        gray = self._get_gray()
        self.add_step("1. Grayscale", gray)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        self.add_step("2. Gaussian Blur", blurred)
        
        edges = cv2.Canny(blurred, threshold1, threshold2)
        self.add_step("3. Canny Edges", edges)
        return edges

    # ==========================================
    # 3️⃣ 影像頻率域濾波 (Frequency Domain Filtering)
    # ==========================================
    def _get_frequency_components(self, channel):
        """輔助函式：執行 FFT 並將低頻移至中心"""
        f_transform = np.fft.fft2(channel)
        return np.fft.fftshift(f_transform)

    def _apply_frequency_filter(self, channel, mask):
        """輔助函式：套用頻率遮罩，並執行反傅立葉轉換 (IFFT) 轉回空間域"""
        f_shift = self._get_frequency_components(channel)
        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.abs(np.fft.ifft2(f_ishift))
        return np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))

    def _create_distance_matrix(self, rows, cols):
        """輔助函式：建立頻譜中心到各點的距離矩陣 (用於生成圓形遮罩)"""
        crow, ccol = rows // 2, cols // 2
        u = np.arange(rows)
        v = np.arange(cols)
        V, U = np.meshgrid(v, u)
        return np.sqrt((U - crow)**2 + (V - ccol)**2)

    def frequency_filter(self, filter_type='gaussian', pass_type='low', D0=30, n=2):
        """
        頻率域濾波主函式
        支援：理想(Ideal)、巴特沃斯(Butterworth)、高斯(Gaussian) 的低通與高通濾波。
        """
        rows, cols = self.img.shape[:2]
        D = self._create_distance_matrix(rows, cols)
        mask = np.zeros_like(D)

        # 根據選擇的濾波器類型與高/低通，計算對應的遮罩 (Mask)
        if filter_type == 'ideal':
            if pass_type == 'low': mask[D <= D0] = 1
            else: mask[D > D0] = 1
        elif filter_type == 'butterworth':
            D_safe = np.where(D == 0, 1e-5, D) # 避免除以零錯誤
            if pass_type == 'low': mask = 1 / (1 + (D_safe / D0)**(2 * n))
            else: mask = 1 / (1 + (D0 / D_safe)**(2 * n))
        elif filter_type == 'gaussian':
            if pass_type == 'low': mask = np.exp(-(D**2) / (2 * (D0**2)))
            else: mask = 1 - np.exp(-(D**2) / (2 * (D0**2)))

        # 執行濾波：若是彩色影像，需將 B, G, R 拆開獨立處理再合併
        if self.is_color:
            b, g, r = cv2.split(self.img)
            b_filtered = self._apply_frequency_filter(b, mask)
            g_filtered = self._apply_frequency_filter(g, mask)
            r_filtered = self._apply_frequency_filter(r, mask)
            return cv2.merge([b_filtered, g_filtered, r_filtered])
        else:
            return self._apply_frequency_filter(self.img, mask)

    # ==========================================
    # 🌟 週期性干擾移除 (陷波濾波器 Notch Filter)
    # ==========================================
    def notch_reject_filter(self, D0_u=30, D0_v=30, u0=50, v0=50, n=2):
        """
        巴特沃斯陷波阻帶濾波器 (Butterworth Notch Reject Filter)
        🌟 終極版：支援「橢圓形」阻帶，並使用標準巴特沃斯乘積公式，完美消除條紋光且無振鈴效應
        D0_u: 水平半徑 (Width)
        D0_v: 垂直半徑 (Height)
        """
        rows, cols = self.img.shape[:2]
        crow, ccol = rows // 2, cols // 2
        u = np.arange(rows)
        v = np.arange(cols)
        V, U = np.meshgrid(v, u)
        
        # 避免半徑為 0 導致除以零錯誤
        D0_u = max(1e-5, D0_u)
        D0_v = max(1e-5, D0_v)
        
        # 計算到兩個對稱干擾點的「橢圓正規化距離」
        # 當距離剛好在橢圓邊界上時，D1 或 D2 會等於 1
        D1 = np.sqrt(((U - crow - u0)**2) / (D0_u**2) + ((V - ccol - v0)**2) / (D0_v**2))
        D2 = np.sqrt(((U - crow + u0)**2) / (D0_u**2) + ((V - ccol + v0)**2) / (D0_v**2))
        
        # 避免除以零
        D1_safe = np.where(D1 == 0, 1e-5, D1)
        D2_safe = np.where(D2 == 0, 1e-5, D2)
        
        # 🌟 專業優化：分別計算兩個點的巴特沃斯遮罩再相乘
        # 這樣能確保頻譜能量的衰減完全符合巴特沃斯的平滑特性，徹底消除振鈴效應
        H1 = 1 / (1 + (1 / D1_safe)**(2 * n))
        H2 = 1 / (1 + (1 / D2_safe)**(2 * n))
        mask = H1 * H2
        
        if self.is_color:
            b, g, r = cv2.split(self.img)
            b_filtered = self._apply_frequency_filter(b, mask)
            g_filtered = self._apply_frequency_filter(g, mask)
            r_filtered = self._apply_frequency_filter(r, mask)
            return cv2.merge([b_filtered, g_filtered, r_filtered])
        else:
            return self._apply_frequency_filter(self.img, mask)


    # ==========================================
    # 🌟 特徵檢測 (霍夫變換 Hough Transform)
    # ==========================================
    def hough_lines(self, threshold=100):
        """霍夫線檢測：找出影像中的直線特徵 (加入步驟紀錄)"""
        gray = self._get_gray()
        self.add_step("1. Grayscale", gray)
        
        # 先用 Canny 找出邊緣
        edges = cv2.Canny(gray, 50, 150)
        self.add_step("2. Canny Edges", edges)
        
        # 執行機率霍夫線變換
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength=50, maxLineGap=10)
        
        # 在原圖上繪製結果 (轉為彩色以便畫綠線)
        result = self.img.copy()
        if not self.is_color:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        self.add_step("3. Draw Lines", result)
        return result

    def hough_circles(self, min_dist=20, param1=50, param2=30, min_radius=0, max_radius=0):
        """霍夫圓檢測：找出影像中的圓形特徵 (加入步驟紀錄)"""
        gray = self._get_gray()
        self.add_step("1. Grayscale", gray)
        
        # 圓檢測對雜訊敏感，先做輕微模糊
        blurred = cv2.medianBlur(gray, 5)
        self.add_step("2. Median Blur", blurred)
        
        # 執行霍夫圓變換
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, min_dist,
                                   param1=param1, param2=param2, 
                                   minRadius=min_radius, maxRadius=max_radius)
        
        result = self.img.copy()
        if not self.is_color:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # 畫出外圓 (綠色)
                cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # 畫出圓心 (紅色)
                cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)
                
        self.add_step("3. Draw Circles", result)
        return result
