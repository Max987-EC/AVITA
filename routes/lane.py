# 車道線偵測路由處理 (Lane Detection Routes)
# 此模組負責影片上傳、逐幀處理、格式轉碼及記憶體串流發送。

import os
import uuid
import cv2
import io
from flask import Blueprint, render_template, request, send_file
from moviepy import VideoFileClip
from lane_detection import LaneDetector

lane_bp = Blueprint('lane', __name__)

@lane_bp.route('/tool/lane-detection', methods=['GET', 'POST'])
def lane_detection():
    """主路由處理器：負責渲染頁面或啟動影片偵測程序"""
    if request.method == 'GET':
        return render_template('lane_detection.html')

    # 1. 驗證影片上傳
    if 'video' not in request.files:
        return "沒有上傳影片", 400
    video_file = request.files['video']
    
    # 2. 建立唯一的臨時路徑，避免併發衝突
    temp_id = uuid.uuid4().hex
    input_path = f"temp_in_{temp_id}.mp4"          
    opencv_output_path = f"temp_cv2_{temp_id}.mp4" 
    web_output_path = f"temp_web_{temp_id}.mp4"    
    
    video_file.save(input_path)

    # 3. 初始化偵測器並讀取影片屬性
    detector = LaneDetector()
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 4. 使用 OpenCV 逐幀處理
    # 注意：輸出的寬度設為 width * 2 以容納「原圖 + 處理圖」對比
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(opencv_output_path, fourcc, fps, (width * 2, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break 
        # 處理每一幀畫面
        processed_frame = detector.process_frame(frame)
        out.write(processed_frame)

    cap.release()
    out.release()
    
    # 5. 使用 MoviePy 進行 Web 格式轉碼 (OpenCV 生成的 MP4 在瀏覽器相容性較差)
    try:
        clip = VideoFileClip(opencv_output_path)
        # 轉換為 H.264 編碼，移除音軌以加速處理
        clip.write_videofile(web_output_path, codec="libx264", audio=False, logger=None)
        clip.close()
    except Exception as e:
        return f"影片轉碼失敗: {str(e)}", 500

    # 6. 核心魔法：將最終影片讀入記憶體緩衝區 (BytesIO)
    # 這樣可以立即刪除暫存檔，且不需要維護靜態輸出目錄
    return_data = io.BytesIO()
    with open(web_output_path, 'rb') as f:
        return_data.write(f.read())
    return_data.seek(0) # 指標歸零

    # 7. 閱後即焚：徹底清理硬碟上的所有暫存檔案
    if os.path.exists(input_path): os.remove(input_path)
    if os.path.exists(opencv_output_path): os.remove(opencv_output_path)
    if os.path.exists(web_output_path): os.remove(web_output_path)

    # 8. 發送影片串流回前端
    return send_file(return_data, mimetype='video/mp4', download_name='result.mp4')
