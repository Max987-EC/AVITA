# 頻率域濾波模組 (Frequency Domain Filtering Module)
# 使用快速傅立葉轉換 (FFT) 將影像轉換至頻率域進行處理。

import cv2
import numpy as np

class FrequencyMixin:
    """
    頻率域處理混入類別 (Frequency Domain Processing Mixin):
    包含理想、巴特沃斯、高斯濾波器，以及帶阻與 Notch 濾波。
    """
    
    # ==========================================
    # 🎨 輔助工具：頻譜視覺化 (Spectrum Visualization)
    # ==========================================
    def _get_spectrum_vis(self, f_shift):
        """將複數頻譜轉換為可視化的灰階幅度譜"""
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        return np.uint8(cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX))

    def _apply_frequency_filter(self, channel, mask, is_gray=False):
        """內部工具：對單一通道套用頻率域遮罩"""
        # 1. 執行二維 FFT 並平移零頻率至中心
        f_transform = np.fft.fft2(channel)
        f_shift = np.fft.fftshift(f_transform)
        
        # 2. 在頻率域套用遮罩 (濾波器函數)
        f_shift_filtered = f_shift * mask
        
        # --- 視覺化步驟 (僅對灰階圖記錄以供教學) ---
        if is_gray:
            # 顯示濾波後的頻譜
            filtered_spectrum_vis = self._get_spectrum_vis(f_shift_filtered)
            self.steps.append(("Step 2: 濾波後的頻譜 (Filtered Spectrum)", filtered_spectrum_vis))
            
            # 顯示對應的空間域卷積核 (PSF): 對遮罩做反 FFT
            spatial_kernel = np.fft.fftshift(np.abs(np.fft.ifft2(np.fft.ifftshift(mask))))
            spatial_kernel_vis = np.uint8(cv2.normalize(np.log(spatial_kernel + 1e-5), None, 0, 255, cv2.NORM_MINMAX))
            self.steps.append(("Step 3: 對應的空間域卷積核 (Spatial Kernel / PSF)", spatial_kernel_vis))

        # 3. 反向平移並執行反向 FFT 回到空間域
        img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift_filtered)))
        return np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))

    def frequency_filter(self, filter_type='gaussian', pass_type='low', D0=30, n=2, W=10, a=0.5, b=1.5):
        """
        標準頻率域濾波器 (Standard Frequency Filters):
        支援低通 (模糊)、高通 (銳化)、帶阻/帶通、高頻強調。
        """
        rows, cols = self.img.shape[:2]
        # 建立頻率平面座標 (中心點為原點)
        U, V = np.meshgrid(np.arange(cols), np.arange(rows))
        D = np.sqrt((U - cols//2)**2 + (V - rows//2)**2)
        mask = np.zeros_like(D)

        # 1. 低通與高通濾波 (Low-pass / High-pass)
        if pass_type in ['low', 'high']:
            if filter_type == 'ideal':
                mask[D <= D0] = 1 if pass_type == 'low' else 0
                mask[D > D0] = 0 if pass_type == 'low' else 1
            elif filter_type == 'butterworth':
                D_safe = np.where(D == 0, 1e-5, D)
                mask = 1 / (1 + (D_safe / D0)**(2 * n)) if pass_type == 'low' else 1 / (1 + (D0 / D_safe)**(2 * n))
            elif filter_type == 'gaussian':
                mask = np.exp(-(D**2) / (2 * (D0**2)))
                if pass_type == 'high': mask = 1 - mask

        # 2. 帶阻與帶通濾波 (Bandreject / Bandpass)
        elif pass_type in ['bandreject', 'bandpass']:
            D_safe = np.where(D == 0, 1e-5, D)
            if filter_type == 'ideal':
                mask = np.ones_like(D)
                mask[(D >= D0 - W/2) & (D <= D0 + W/2)] = 0
            elif filter_type == 'butterworth':
                mask = 1 / (1 + ((D * W) / (D**2 - D0**2 + 1e-5))**(2 * n))
            elif filter_type == 'gaussian':
                mask = 1 - np.exp(-((D**2 - D0**2)**2) / (D * W + 1e-5)**2)
            
            if pass_type == 'bandpass':
                mask = 1 - mask

        # 3. 高頻強調濾波 (High-Frequency Emphasis)
        # 公式: H_emphasis = a + b * H_highpass
        elif pass_type == 'emphasis':
            D_safe = np.where(D == 0, 1e-5, D)
            high_pass_mask = 1 / (1 + (D0 / D_safe)**(2 * n))
            mask = a + b * high_pass_mask

        # 視覺化：記錄濾波器遮罩影像
        mask_vis = np.uint8(cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX))
        self.steps.append(("Step 1: 頻率域遮罩 (Frequency Mask)", mask_vis))

        if self.is_color:
            # 彩色影像分別對三個通道濾波再合併
            result = cv2.merge([self._apply_frequency_filter(c, mask, is_gray=False) for c in cv2.split(self.img)])
            self.steps.append(("Step 2: 頻率域濾波最終結果", result))
            return result
        
        # 灰階影像則展示完整的頻譜演進過程
        result = self._apply_frequency_filter(self.img, mask, is_gray=True)
        self.steps.append(("Step 4: 頻率域濾波最終結果", result))
        return result

    def notch_reject_filter(self, D0_u=30, D0_v=30, u0=50, v0=50, n=2, notch_type='reject'):
        """
        陷波濾波器 (Notch Filter):
        用於消除影像中特定頻率的週期性雜訊。
        """
        rows, cols = self.img.shape[:2]
        U, V = np.meshgrid(np.arange(cols), np.arange(rows))
        
        D0_u, D0_v = max(1e-5, D0_u), max(1e-5, D0_v)
        # 計算到正負對稱陷波點的距離
        D1 = np.sqrt(((V - rows//2 - u0)**2) / (D0_u**2) + ((U - cols//2 - v0)**2) / (D0_v**2))
        D2 = np.sqrt(((V - rows//2 + u0)**2) / (D0_u**2) + ((U - cols//2 + v0)**2) / (D0_v**2))
        
        # 使用巴特沃斯陷波公式
        mask = (1 / (1 + (1 / np.where(D1 == 0, 1e-5, D1))**(2 * n))) * \
               (1 / (1 + (1 / np.where(D2 == 0, 1e-5, D2))**(2 * n)))
        
        if notch_type == 'pass':
            mask = 1 - mask
            
        mask_vis = np.uint8(cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX))
        self.steps.append(("Step 1: Notch 遮罩 (Notch Mask)", mask_vis))
        
        if self.is_color:
            result = cv2.merge([self._apply_frequency_filter(c, mask, is_gray=False) for c in cv2.split(self.img)])
            self.steps.append(("Step 2: 頻率域濾波最終結果", result))
            return result
            
        result = self._apply_frequency_filter(self.img, mask, is_gray=True)
        self.steps.append(("Step 4: 頻率域濾波最終結果", result))
        return result
