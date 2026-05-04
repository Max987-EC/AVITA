# 負責首頁大廳

from flask import Blueprint, render_template

# 建立 Blueprint，命名為 'main'
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """渲染系統首頁"""
    return render_template('index.html')
