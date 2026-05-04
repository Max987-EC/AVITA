# 負責車道線偵測

import os
import uuid
import cv2
from flask import Blueprint, render_template, request, send_file
from moviepy import VideoFileClip
from lane_detector import LaneDetector

lane_bp = Blueprint('lane', __name__)

@lane_bp.route('/tool/lane-detection', methods=['GET', 'POST'])
def lane_detection():
    if request.method == 'GET':
        return render_template('lane_detection.html')

    if 'video' not in request.files:
        return "沒有上傳影片", 400
    video_file = request.files['video']
    
    temp_id = uuid.uuid4().hex
    input_path = f"temp_in_{temp_id}.mp4"          
    opencv_output_path = f"temp_cv2_{temp_id}.mp4" 
    web_output_path = f"temp_web_{temp_id}.mp4"    
    
    video_file.save(input_path)

    detector = LaneDetector()
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(opencv_output_path, fourcc, fps, (width * 2, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break 
        processed_frame = detector.process_frame(frame)
        out.write(processed_frame)

    cap.release()
    out.release()
    
    try:
        clip = VideoFileClip(opencv_output_path)
        clip.write_videofile(web_output_path, codec="libx264", audio=False, logger=None)
        clip.close()
    except Exception as e:
        return f"影片轉碼失敗: {str(e)}", 500

    if os.path.exists(input_path): os.remove(input_path)
    if os.path.exists(opencv_output_path): os.remove(opencv_output_path)

    return send_file(web_output_path, mimetype='video/mp4')
