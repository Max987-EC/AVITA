# 影像處理路由與算子映射 (Image Processing Routes & Operation Mapping)
# 此模組負責接收前端影像處理請求，並將其映射至對應的 ImageProcessor 方法。

import cv2
import numpy as np
import base64
import json
from flask import Blueprint, render_template, request, jsonify
from image_processing import ImageProcessor, apply_operation

image_bp = Blueprint('image', __name__)

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
                
                # 讀取前端傳來的啟用狀態 (預設為 True)
                is_enabled = node.get('enabled', True)
                
                # 紀錄該節點的輸入影像
                success_in, encoded_in = cv2.imencode('.jpg', current_img)
                node_input_b64 = base64.b64encode(encoded_in).decode('utf-8') if success_in else ""
                
                if is_enabled:
                    # 🟢 若節點啟用，正常執行算子 (使用抽離的 apply_operation)
                    processor = ImageProcessor(current_img)
                    result_img = apply_operation(processor, op, params)
                    # 收集該算子產生的所有中間步驟
                    steps_b64 = [{"name": name, "image": base64.b64encode(cv2.imencode('.jpg', step_img)[1]).decode('utf-8')} for name, step_img in processor.steps]
                else:
                    # 🔴 若節點被停用 (Bypass)，直接將輸入當作輸出，不進行處理
                    result_img = current_img.copy()
                    steps_b64 = [{"name": "⚠️ 此步驟已停用 (Skipped)", "image": node_input_b64}]
                
                # 紀錄該節點的輸出影像
                success_out, encoded_out = cv2.imencode('.jpg', result_img)
                node_output_b64 = base64.b64encode(encoded_out).decode('utf-8') if success_out else ""
                
                # 打包節點數據
                pipeline_results.append({
                    'node_idx': i,
                    'operation_name': node['name'],
                    'input_img': node_input_b64,
                    'output_img': node_output_b64,
                    'steps': steps_b64,
                    'is_enabled': is_enabled,  # 將狀態回傳給前端
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
            
            # 直接將表單內容視為參數傳入執行器 (使用抽離的 apply_operation)
            result_img = apply_operation(processor, process_type, request.form)

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
