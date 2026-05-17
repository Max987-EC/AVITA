# 負責頻率域濾波

import cv2
import numpy as np

class FrequencyMixin:
    
    # ==========================================
    # 🎨 輔助工具：產生頻譜圖的視覺化影像
    # ==========================================
    def _get_spectrum_vis(self, f_shift):
        """將複數頻譜轉換為可視化的灰階影像"""
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        return np.uint8(cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX))

    def _apply_frequency_filter(self, channel, mask, is_gray=False):
        # 1. 傅立葉轉換並平移到中心
        f_transform = np.fft.fft2(channel)
        f_shift = np.fft.fftshift(f_transform)
        
        # 2. 套用遮罩
        f_shift_filtered = f_shift * mask
        
        # --- 🎨 視覺化步驟 (僅在單通道或灰階圖時記錄，避免彩色圖重複記錄三次) ---
        if is_gray:
            # Step 1: 顯示濾波後的頻譜
            filtered_spectrum_vis = self._get_spectrum_vis(f_shift_filtered)
            self.steps.append(("Step 2: 濾波後的頻譜 (Filtered Spectrum)", filtered_spectrum_vis))
            
            # Step 2: 顯示空間域卷積核 (將 Mask 做反傅立葉轉換)
            # 為了讓核的中心在中間，我們對 mask 做 ifftshift 再 ifft2，然後再 shift 回來觀察
            spatial_kernel = np.fft.fftshift(np.abs(np.fft.ifft2(np.fft.ifftshift(mask))))
            # 使用對數轉換讓微小的波紋(Ringing)更明顯
            spatial_kernel_vis = np.uint8(cv2.normalize(np.log(spatial_kernel + 1e-5), None, 0, 255, cv2.NORM_MINMAX))
            self.steps.append(("Step 3: 對應的空間域卷積核 (Spatial Kernel / PSF)", spatial_kernel_vis))
        # ----------------------------------------------------------------

        # 3. 反傅立葉轉換回空間域
        img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift_filtered)))
        return np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))

    def frequency_filter(self, filter_type='gaussian', pass_type='low', D0=30, n=2, W=10, a=0.5, b=1.5):
        rows, cols = self.img.shape[:2]
        U, V = np.meshgrid(np.arange(cols), np.arange(rows))
        D = np.sqrt((U - cols//2)**2 + (V - rows//2)**2)
        mask = np.zeros_like(D)

        # 1. 理想、巴特沃斯、高斯 (低通/高通)
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

        # 2. 帶阻 / 帶通 (Bandreject / Bandpass)
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

        # 3. 高頻強調 (High-Frequency Emphasis)
        elif pass_type == 'emphasis':
            D_safe = np.where(D == 0, 1e-5, D)
            high_pass_mask = 1 / (1 + (D0 / D_safe)**(2 * n))
            mask = a + b * high_pass_mask

        # 🎨 視覺化：記錄頻率域遮罩
        mask_vis = np.uint8(cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX))
        self.steps.append(("Step 1: 頻率域遮罩 (Frequency Mask)", mask_vis))

        if self.is_color:
            # 彩色影像不顯示單通道的頻譜步驟，以免畫面太雜亂
            result = cv2.merge([self._apply_frequency_filter(c, mask, is_gray=False) for c in cv2.split(self.img)])
            self.steps.append(("Step 2: 頻率域濾波最終結果", result))
            return result
        
        # 灰階影像則顯示完整的頻譜與卷積核步驟
        result = self._apply_frequency_filter(self.img, mask, is_gray=True)
        self.steps.append(("Step 4: 頻率域濾波最終結果", result))
        return result

    def notch_reject_filter(self, D0_u=30, D0_v=30, u0=50, v0=50, n=2, notch_type='reject'):
        rows, cols = self.img.shape[:2]
        U, V = np.meshgrid(np.arange(cols), np.arange(rows))
        
        D0_u, D0_v = max(1e-5, D0_u), max(1e-5, D0_v)
        D1 = np.sqrt(((V - rows//2 - u0)**2) / (D0_u**2) + ((U - cols//2 - v0)**2) / (D0_v**2))
        D2 = np.sqrt(((V - rows//2 + u0)**2) / (D0_u**2) + ((U - cols//2 + v0)**2) / (D0_v**2))
        
        mask = (1 / (1 + (1 / np.where(D1 == 0, 1e-5, D1))**(2 * n))) * \
               (1 / (1 + (1 / np.where(D2 == 0, 1e-5, D2))**(2 * n)))
        
        if notch_type == 'pass':
            mask = 1 - mask
            
        # 🎨 視覺化：記錄 Notch 遮罩
        mask_vis = np.uint8(cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX))
        self.steps.append(("Step 1: Notch 遮罩 (Notch Mask)", mask_vis))
        
        if self.is_color:
            result = cv2.merge([self._apply_frequency_filter(c, mask, is_gray=False) for c in cv2.split(self.img)])
            self.steps.append(("Step 2: 頻率域濾波最終結果", result))
            return result
            
        result = self._apply_frequency_filter(self.img, mask, is_gray=True)
        self.steps.append(("Step 4: 頻率域濾波最終結果", result))
        return result
