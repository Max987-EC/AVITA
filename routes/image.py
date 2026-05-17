# 影像處理路由與算子映射 (Image Processing Routes & Operation Mapping)
# 此模組負責接收前端影像處理請求，並將其映射至對應的 ImageProcessor 方法。

import cv2
import numpy as np
import base64
import json
from flask import Blueprint, render_template, request, jsonify
from image_processing import ImageProcessor

image_bp = Blueprint('image', __name__)

# ==========================================
# 🛠️ 核心工具：統一算子執行器 (Unified Operation Executor)
# ==========================================
def _apply_operation(processor, op, params):
    """
    根據算子名稱與參數字典，執行對應的影像處理器方法。
    這是一個中央映射器，確保「單步模式」與「管線模式」共用同一套邏輯。
    
    參數:
    - processor: ImageProcessor 實例
    - op: 算子識別字串 (如 'binarize')
    - params: 字典格式的參數集 (如 {'threshold': 127})
    """
    
    # --- 1. 基本強度轉換與點處理 ---
    if op == 'binarize':
        return processor.binarize(
            int(params.get('threshold', 127)), 
            params.get('thresh_type', 'binary'), 
            str(params.get('use_otsu', 'false')).lower() == 'true'
        )
    elif op == 'adaptive_threshold':
        return processor.adaptive_threshold(
            max_value=255,
            adaptive_method=params.get('adaptive_method', 'gaussian'),
            thresh_type=params.get('thresh_type', 'binary'),
            block_size=int(params.get('block_size', 11)),
            C=int(params.get('C', 2))
        )
    elif op == 'moving_average':
        return processor.moving_average_threshold(
            n=int(params.get('ma_n', 20)),
            k=float(params.get('ma_k', 0.5))
        )
    elif op == 'edge_improved_threshold':
        return processor.edge_improved_threshold()
    elif op == 'linear':
        return processor.linear_transform(float(params.get('alpha', 1.0)), int(params.get('beta', 0)))
    elif op == 'clahe':
        return processor.clahe_equalization(float(params.get('clip_limit', 2.0)), int(params.get('tile_grid_size', 8)))
    elif op == 'negative':
        return processor.negative_transform()
    elif op == 'log':
        return processor.log_transform(float(params.get('c', 1.0)))
    elif op == 'power_law':
        return processor.power_law_transform(float(params.get('gamma', 1.0)), float(params.get('c', 1.0)))
    elif op == 'equalization':
        return processor.histogram_equalization()
        
    # --- 2. 形態學運算 (基礎與邊界) ---
    # 解析前端傳來的自訂 Kernel (可能是 JSON 字串)
    custom_kernel_raw = params.get('custom_kernel')
    custom_kernel = None
    if custom_kernel_raw:
        if isinstance(custom_kernel_raw, str):
            try:
                custom_kernel = json.loads(custom_kernel_raw)
            except json.JSONDecodeError:
                custom_kernel = None
        else:
            custom_kernel = custom_kernel_raw

    if op == 'erosion':
        return processor.erosion(params.get('shape', 'rect'), int(params.get('ksize', 3)), int(params.get('iterations', 1)), custom_kernel)
    elif op == 'dilation':
        return processor.dilation(params.get('shape', 'rect'), int(params.get('ksize', 3)), int(params.get('iterations', 1)), custom_kernel)
    elif op == 'opening':
        return processor.opening(params.get('shape', 'rect'), int(params.get('ksize', 3)), int(params.get('iterations', 1)), custom_kernel)
    elif op == 'closing':
        return processor.closing(params.get('shape', 'rect'), int(params.get('ksize', 3)), int(params.get('iterations', 1)), custom_kernel)
    elif op == 'tophat':
        return processor.tophat(params.get('shape', 'rect'), int(params.get('ksize', 15)), custom_kernel)
    elif op == 'blackhat':
        return processor.blackhat(params.get('shape', 'rect'), int(params.get('ksize', 15)), custom_kernel)
    elif op == 'morph_gradient':
        return processor.morph_gradient(params.get('shape', 'rect'), int(params.get('ksize', 3)), custom_kernel)
    elif op == 'boundary_extraction':
        return processor.boundary_extraction(params.get('shape', 'rect'), int(params.get('ksize', 3)), custom_kernel)
    elif op == 'morph_smoothing':
        return processor.morph_smoothing(
            shape=params.get('shape', 'ellipse'),
            ksize=int(params.get('ksize', 5)),
            mode=params.get('smooth_mode', 'open_close'),
            custom_kernel=custom_kernel
        )
        
    # --- 3. 進階形態學與分析量測 ---
    elif op == 'hole_filling':
        return processor.hole_filling()
    elif op == 'hit_or_miss':
        return processor.hit_or_miss()
    elif op == 'connected_components':
        min_area = int(params.get('min_area')) if params.get('min_area') else 0
        max_area = int(params.get('max_area')) if params.get('max_area') else 9999999
        connectivity = int(params.get('connectivity', 8))
        return processor.connected_components(min_area=min_area, max_area=max_area, connectivity=connectivity)
        
    # --- 4. 進階幾何特徵與次像素量測 ---
    elif op == 'advanced_features':
        return processor.advanced_features(
            min_area=int(params.get('adv_min_area', 100)),
            draw_bounding_rect=str(params.get('draw_bounding_rect', 'false')).lower() == 'true',
            draw_min_rect=str(params.get('draw_min_rect', 'false')).lower() == 'true',
            draw_circle=str(params.get('draw_circle', 'false')).lower() == 'true',
            draw_ellipse=str(params.get('draw_ellipse', 'false')).lower() == 'true',
            draw_hull=str(params.get('draw_hull', 'false')).lower() == 'true',
            calc_gray=str(params.get('calc_gray', 'false')).lower() == 'true',
            calc_perimeter=str(params.get('calc_perimeter', 'false')).lower() == 'true',
            calc_shape=str(params.get('calc_shape', 'false')).lower() == 'true'
        )
    elif op == 'subpixel_corners':
        return processor.subpixel_corners(
            max_corners=int(params.get('max_corners', 100)),
            quality=float(params.get('quality', 0.01)),
            min_dist=int(params.get('min_dist', 10))
        )
    elif op == 'subpixel_contours':
        return processor.subpixel_contours(
            threshold=int(params.get('sub_threshold', 127)),
            upscale_factor=int(params.get('upscale_factor', 4))
        )
    elif op == 'fuzzy_measurement':
        return processor.fuzzy_measurement(
            min_val=int(params.get('fmv_min', 40)),
            max_val=int(params.get('fmv_max', 120))
        )

    # --- 5. 影像分割演算法 ---
    elif op == 'region_growing':
        seed_x_val = params.get('seed_x')
        seed_y_val = params.get('seed_y')
        seed_x = int(seed_x_val) if seed_x_val else None
        seed_y = int(seed_y_val) if seed_y_val else None
        tolerance = int(params.get('tolerance')) if params.get('tolerance') else 20
        return processor.region_growing(seed_x=seed_x, seed_y=seed_y, tolerance=tolerance)
    elif op == 'region_split_merge':
        return processor.region_split_merge(
            std_threshold=float(params.get('std_threshold', 15.0)),
            min_size=int(params.get('min_size', 8))
        )

    # --- 6. 空間域濾波器 ---
    elif op == 'mean':
        return processor.mean_filter(int(params.get('kernel_size', 3)))
    elif op == 'gaussian':
        return processor.gaussian_filter(int(params.get('kernel_size', 3)), float(params.get('sigma', 1.0)))
    elif op == 'median':
        return processor.median_filter(int(params.get('kernel_size', 3)))
    elif op == 'bilateral':
        return processor.bilateral_filter(
            int(params.get('bilateral_d', 9)), 
            float(params.get('bilateral_sigma_color', 75.0)), 
            float(params.get('bilateral_sigma_space', 75.0))
        )
    elif op == 'sharpen':
        return processor.sharpen_filter(float(params.get('sharpen_amount', 1.0)))
        
    # --- 7. 邊緣偵測 ---
    elif op == 'roberts':
        return processor.roberts_filter(params.get('edge_direction', 'both'))
    elif op == 'prewitt':
        return processor.prewitt_filter(params.get('edge_direction', 'both'))
    elif op == 'sobel':
        return processor.sobel_filter(int(params.get('kernel_size', 3)), params.get('edge_direction', 'both'))
    elif op == 'laplacian':
        return processor.laplacian_filter(int(params.get('kernel_size', 3)))
    elif op == 'log_edge':
        return processor.log_filter(int(params.get('kernel_size', 3)), float(params.get('sigma', 1.0)))
    elif op == 'canny':
        return processor.canny_filter(
            int(params.get('threshold1', 50)), int(params.get('threshold2', 150)), 
            int(params.get('canny_blur_ksize', 5)), float(params.get('canny_blur_sigma', 1.4))
        )
    elif op == 'canny_custom':
        return processor.canny_custom_filter(
            int(params.get('threshold1', 50)), int(params.get('threshold2', 150)), 
            int(params.get('canny_blur_ksize', 5)), float(params.get('canny_blur_sigma', 1.4))
        )
        
    # --- 8. 頻率域濾波 ---
    elif op == 'notch_reject':
        return processor.notch_reject_filter(
            float(params.get('notch_d0_u', 30.0)), float(params.get('notch_d0_v', 30.0)), 
            int(params.get('u0', 50)), int(params.get('v0', 50)), int(params.get('n', 2)), 
            params.get('notch_type', 'reject')
        )
    elif op.startswith('freq_'):
        parts = op.split('_')
        return processor.frequency_filter(
            parts[1], parts[2], float(params.get('D0', 30.0)), int(params.get('n', 2)), 
            float(params.get('freq_W', 10.0)), float(params.get('freq_a', 0.5)), float(params.get('freq_b', 1.5))
        )
        
    # --- 9. 特徵檢測 (Hough Transform) ---
    elif op == 'hough_lines_standard':
        return processor.hough_lines_standard(
            int(params.get('hough_threshold', 100)), float(params.get('hough_rho', 1.0)), float(params.get('hough_theta', 1.0)), 
            int(params.get('hough_canny_th1', 50)), int(params.get('hough_canny_th2', 150)), int(params.get('hough_blur_ksize', 5))
        )
    elif op == 'hough_lines_p':
        return processor.hough_lines_p(
            int(params.get('hough_threshold', 100)), int(params.get('hough_min_line_length', 50)), int(params.get('hough_max_line_gap', 10)), 
            float(params.get('hough_rho', 1.0)), float(params.get('hough_theta', 1.0)), 
            int(params.get('hough_canny_th1', 50)), int(params.get('hough_canny_th2', 150)), int(params.get('hough_blur_ksize', 5))
        )
    elif op == 'hough_circles':
        return processor.hough_circles(
            float(params.get('hough_dp', 1.0)), int(params.get('hough_min_dist', 20)), 
            int(params.get('hough_param1', 50)), int(params.get('hough_param2', 30)), 
            int(params.get('hough_min_radius', 0)), int(params.get('hough_max_radius', 0)), int(params.get('hough_blur_ksize', 5))
        )
        
    return processor.img


# ==========================================
# 🌐 API 路由主入口 (API Route Entry Point)
# ==========================================
@image_bp.route('/tool/image-processing', methods=['GET', 'POST'])
def image_processing():
    """主路由處理器：負責渲染頁面或處理影像計算請求"""
    if request.method == 'GET':
        return render_template('image_processing.html')

    if request.method == 'POST':
        # 1. 驗證圖片上傳狀態
        if 'image' not in request.files:
            return jsonify({"error": "未偵測到上傳圖片"}), 400
            
        file = request.files['image']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_array is None:
            return jsonify({"error": "影像格式解析失敗"}), 400

        # 2. 根據 pipeline_sequence 判斷執行模式
        pipeline_str = request.form.get('pipeline_sequence')
        
        if pipeline_str:
            # ------------------------------------------
            # 🌟 [模式 A] 管線堆疊 (Pipeline Mode)
            # ------------------------------------------
            pipeline = json.loads(pipeline_str)
            pipeline_results = []
            current_img = img_array.copy()
            
            # 編碼全局原圖
            success_orig, encoded_orig = cv2.imencode('.jpg', current_img)
            global_input_b64 = base64.b64encode(encoded_orig).decode('utf-8') if success_orig else ""
            
            # 依序執行堆疊中的算子
            for i, node in enumerate(pipeline):
                op = node['type']
                params = node.get('params', {})
                
                # 紀錄該節點的輸入影像
                success_in, encoded_in = cv2.imencode('.jpg', current_img)
                node_input_b64 = base64.b64encode(encoded_in).decode('utf-8') if success_in else ""
                
                processor = ImageProcessor(current_img)
                
                # 呼叫映射器執行算子
                result_img = _apply_operation(processor, op, params)
                
                # 紀錄該節點的輸出影像
                success_out, encoded_out = cv2.imencode('.jpg', result_img)
                node_output_b64 = base64.b64encode(encoded_out).decode('utf-8') if success_out else ""
                
                # 收集該算子產生的所有中間步驟
                steps_b64 = [{"name": name, "image": base64.b64encode(cv2.imencode('.jpg', step_img)[1]).decode('utf-8')} for name, step_img in processor.steps]
                
                # 打包節點數據
                pipeline_results.append({
                    'node_idx': i,
                    'operation_name': node['name'],
                    'input_img': node_input_b64,
                    'output_img': node_output_b64,
                    'steps': steps_b64,
                    'original_histogram': ImageProcessor.generate_histogram_base64(current_img),
                    'processed_histogram': ImageProcessor.generate_histogram_base64(result_img),
                    'original_spectrum': ImageProcessor.generate_spectrum_base64(current_img),
                    'processed_spectrum': ImageProcessor.generate_spectrum_base64(result_img)
                })
                
                # 更新目前影像，傳遞給下一層
                current_img = result_img
            
            # 編碼全局最終輸出
            success_final, encoded_final = cv2.imencode('.jpg', current_img)
            global_output_b64 = base64.b64encode(encoded_final).decode('utf-8') if success_final else ""
            
            return jsonify({
                'mode': 'stack',
                'global_input': global_input_b64,
                'global_output': global_output_b64,
                'nodes_data': pipeline_results,
                'original_histogram': ImageProcessor.generate_histogram_base64(img_array),
                'processed_histogram': ImageProcessor.generate_histogram_base64(current_img),
                'original_spectrum': ImageProcessor.generate_spectrum_base64(img_array),
                'processed_spectrum': ImageProcessor.generate_spectrum_base64(current_img)
            })

        else:
            # ------------------------------------------
            # 🌟 [模式 B] 單步分析 (Single Mode)
            # ------------------------------------------
            process_type = request.form.get('process_type', 'negative') 
            
            processor = ImageProcessor(img_array)
            
            # 直接將表單內容視為參數傳入執行器
            result_img = _apply_operation(processor, process_type, request.form)

            # 編碼原圖與結果圖
            success_orig, encoded_orig = cv2.imencode('.jpg', img_array)
            orig_b64 = base64.b64encode(encoded_orig).decode('utf-8') if success_orig else ""

            success, encoded_img = cv2.imencode('.jpg', result_img)
            if not success:
                return jsonify({"error": "影像編碼失敗"}), 500
            processed_b64 = base64.b64encode(encoded_img).decode('utf-8')

            # 收集步驟紀錄
            steps_b64 = [{"name": name, "image": base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8')} for name, img in processor.steps]

            return jsonify({
                "mode": "single",
                "original_image": orig_b64,          
                "processed_image": processed_b64,
                "original_histogram": ImageProcessor.generate_histogram_base64(img_array),
                "processed_histogram": ImageProcessor.generate_histogram_base64(result_img),
                "original_spectrum": ImageProcessor.generate_spectrum_base64(img_array),
                "processed_spectrum": ImageProcessor.generate_spectrum_base64(result_img),
                "steps": steps_b64
            })
