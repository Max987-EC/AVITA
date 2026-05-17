# 核心導覽路由處理 (Main Navigation Routes)
# 此模組負責首頁大廳的渲染。

from flask import Blueprint, render_template

# 建立全站主導覽的 Blueprint
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """
    渲染系統主入口頁面 (AVITA Visual Station)。
    包含各個視覺工具模組的導覽連結與趣味互動特效。
    """
    return render_template('index.html')
