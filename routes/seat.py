# 負責座位排列

from flask import Blueprint, render_template, request, jsonify
from seat_arranger import SeatArranger

seat_bp = Blueprint('seat', __name__)

@seat_bp.route('/tool/seat-arranger', methods=['GET'])
def seat_arranger_route():
    """渲染座位排列系統的網頁"""
    return render_template('seat.html')

@seat_bp.route('/api/seat-arranger/arrange', methods=['POST'])
def arrange_seats_api():
    """接收前端傳來的資料，交由 SeatArranger 處理"""
    data = request.json
    arranger = SeatArranger()
    return jsonify(arranger.handle_request(data))
