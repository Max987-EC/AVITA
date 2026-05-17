# 學生座位排列系統路由 (Seat Arranger Routes)
# 此模組負責座位系統的頁面渲染及排座演算法的 API 交互。

from flask import Blueprint, render_template, request, jsonify
from seat_arranger import SeatArranger

seat_bp = Blueprint('seat', __name__)

@seat_bp.route('/tool/seat-arranger', methods=['GET'])
def seat_arranger_route():
    """
    渲染座位排列系統主頁面。
    提供名單輸入、教室選擇及偏好設定介面。
    """
    return render_template('seat.html')

@seat_bp.route('/api/seat-arranger/arrange', methods=['POST'])
def arrange_seats_api():
    """
    核心排座 API：接收前端 JSON 請求並執行排座演算法。
    
    請求內容 (JSON):
    - students: 學生名單
    - selectedCls: 選中的教室 ID 清單
    - quincunx: 是否啟用梅花座
    - seatMap: 目前已手動鎖定的座位圖
    - customClassrooms: 自定義教室設定
    
    傳回值:
    - 包含渲染後的 HTML 片段與統計數據的 JSON 物件。
    """
    data = request.json
    arranger = SeatArranger()
    
    # 呼叫 SeatArranger 核心類別處理請求邏輯
    return jsonify(arranger.handle_request(data))
