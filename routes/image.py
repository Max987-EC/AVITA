# 負責影像處理

import cv2
import numpy as np
import base64
from flask import Blueprint, render_template, request, jsonify
from image_processing import ImageProcessor

image_bp = Blueprint('image', __name__)

@image_bp.route('/tool/image-processing', methods=['GET', 'POST'])
def image_processing():
    if request.method == 'GET':
        return render_template('image_processing.html')

    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "沒有上傳圖片"}), 400
            
        file = request.files['image']
        process_type = request.form.get('process_type', 'negative') 
        
        # 擷取前端參數：基本強度轉換
        threshold = int(request.form.get('threshold', 127))
        thresh_type = request.form.get('thresh_type', 'binary')
        use_otsu = request.form.get('use_otsu') == 'true'
        
        alpha = float(request.form.get('alpha', 1.0))
        beta = int(request.form.get('beta', 0))
        
        c_val = float(request.form.get('c', 1.0))                        
        gamma = float(request.form.get('gamma', 1.0))                    
        clip_limit = float(request.form.get('clip_limit', 2.0))          
        tile_grid_size = int(request.form.get('tile_grid_size', 8))      
        
        # 擷取前端參數：空間域濾波與邊緣偵測
        kernel_size = int(request.form.get('kernel_size', 3))            
        sigma = float(request.form.get('sigma', 1.0))                    
        
        edge_direction = request.form.get('edge_direction', 'both')
        sharpen_amount = float(request.form.get('sharpen_amount', 1.0))
        bilateral_d = int(request.form.get('bilateral_d', 9))
        bilateral_sigma_color = float(request.form.get('bilateral_sigma_color', 75.0))
        bilateral_sigma_space = float(request.form.get('bilateral_sigma_space', 75.0))
        
        threshold1 = int(request.form.get('threshold1', 50))             
        threshold2 = int(request.form.get('threshold2', 150))            
        canny_blur_ksize = int(request.form.get('canny_blur_ksize', 5))
        canny_blur_sigma = float(request.form.get('canny_blur_sigma', 1.4))
        
        # 擷取前端參數：頻率域
        D0 = float(request.form.get('D0', 30.0))                         
        n_order = int(request.form.get('n', 2))                          
        freq_W = float(request.form.get('freq_W', 10.0))
        freq_a = float(request.form.get('freq_a', 0.5))
        freq_b = float(request.form.get('freq_b', 1.5))

        notch_d0_u = float(request.form.get('notch_d0_u', 30.0))
        notch_d0_v = float(request.form.get('notch_d0_v', 30.0))
        u0 = int(request.form.get('u0', 50))
        v0 = int(request.form.get('v0', 50))
        notch_type = request.form.get('notch_type', 'reject')
        
        # 擷取前端參數：特徵檢測 (新增 rho, theta, dp)
        hough_rho = float(request.form.get('hough_rho', 1.0))
        hough_theta = float(request.form.get('hough_theta', 1.0))
        hough_dp = float(request.form.get('hough_dp', 1.0))
        
        hough_threshold = int(request.form.get('hough_threshold', 100))
        hough_min_line_length = int(request.form.get('hough_min_line_length', 50)) 
        hough_max_line_gap = int(request.form.get('hough_max_line_gap', 10))       
        hough_canny_th1 = int(request.form.get('hough_canny_th1', 50))
        hough_canny_th2 = int(request.form.get('hough_canny_th2', 150))
        
        hough_min_dist = int(request.form.get('hough_min_dist', 20))
        hough_param1 = int(request.form.get('hough_param1', 50))
        hough_param2 = int(request.form.get('hough_param2', 30))
        hough_min_radius = int(request.form.get('hough_min_radius', 0))
        hough_max_radius = int(request.form.get('hough_max_radius', 0))
        hough_blur_ksize = int(request.form.get('hough_blur_ksize', 5))
        
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_array is None:
            return jsonify({"error": "圖片讀取失敗"}), 400

        processor = ImageProcessor(img_array)
        
        # 執行演算法
        if process_type == 'binarize':
            result_img = processor.binarize(threshold, thresh_type, use_otsu)
        elif process_type == 'linear':
            result_img = processor.linear_transform(alpha, beta)
        elif process_type == 'clahe':  
            result_img = processor.clahe_equalization(clip_limit, tile_grid_size)
        elif process_type == 'negative':
            result_img = processor.negative_transform()
        elif process_type == 'log':
            result_img = processor.log_transform(c_val)
        elif process_type == 'power_law':
            result_img = processor.power_law_transform(gamma, c_val)
        elif process_type == 'equalization':
            result_img = processor.histogram_equalization()
        elif process_type == 'mean':
            result_img = processor.mean_filter(kernel_size)
        elif process_type == 'gaussian':
            result_img = processor.gaussian_filter(kernel_size, sigma)
        elif process_type == 'median':
            result_img = processor.median_filter(kernel_size)
        elif process_type == 'bilateral':
            result_img = processor.bilateral_filter(bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
        elif process_type == 'sharpen':
            result_img = processor.sharpen_filter(sharpen_amount)
        elif process_type == 'roberts':
            result_img = processor.roberts_filter(edge_direction)
        elif process_type == 'prewitt':
            result_img = processor.prewitt_filter(edge_direction)
        elif process_type == 'sobel':
            result_img = processor.sobel_filter(kernel_size, edge_direction)
        elif process_type == 'laplacian':
            result_img = processor.laplacian_filter(kernel_size)
        elif process_type == 'log_edge':
            result_img = processor.log_filter(kernel_size, sigma)
        elif process_type == 'canny':
            result_img = processor.canny_filter(threshold1, threshold2, canny_blur_ksize, canny_blur_sigma)
        elif process_type == 'canny_custom':
            result_img = processor.canny_custom_filter(threshold1, threshold2, canny_blur_ksize, canny_blur_sigma)
        elif process_type == 'notch_reject':
            result_img = processor.notch_reject_filter(notch_d0_u, notch_d0_v, u0, v0, n_order, notch_type)
        elif process_type.startswith('freq_'):
            parts = process_type.split('_')
            result_img = processor.frequency_filter(parts[1], parts[2], D0, n_order, freq_W, freq_a, freq_b)
        elif process_type == 'hough_lines_standard':
            result_img = processor.hough_lines_standard(hough_threshold, hough_rho, hough_theta, hough_canny_th1, hough_canny_th2, hough_blur_ksize)
        elif process_type == 'hough_lines_p':
            result_img = processor.hough_lines_p(hough_threshold, hough_min_line_length, hough_max_line_gap, hough_rho, hough_theta, hough_canny_th1, hough_canny_th2, hough_blur_ksize)
        elif process_type == 'hough_circles':
            result_img = processor.hough_circles(hough_dp, hough_min_dist, hough_param1, hough_param2, hough_min_radius, hough_max_radius, hough_blur_ksize)
        else:
            result_img = processor.img

        # 編碼回傳
        success_orig, encoded_orig = cv2.imencode('.jpg', img_array)
        orig_b64 = base64.b64encode(encoded_orig).decode('utf-8') if success_orig else ""

        success, encoded_img = cv2.imencode('.jpg', result_img)
        if not success:
            return jsonify({"error": "影像編碼失敗"}), 500
        processed_b64 = base64.b64encode(encoded_img).decode('utf-8')

        steps_b64 = [{"name": name, "image": base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8')} for name, img in processor.steps]

        return jsonify({
            "original_image": orig_b64,          
            "processed_image": processed_b64,
            "original_histogram": ImageProcessor.generate_histogram_base64(img_array),
            "processed_histogram": ImageProcessor.generate_histogram_base64(result_img),
            "original_spectrum": ImageProcessor.generate_spectrum_base64(img_array),  
            "processed_spectrum": ImageProcessor.generate_spectrum_base64(result_img),
            "steps": steps_b64  
        })
