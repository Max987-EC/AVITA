# 負責將所有模組組裝起來

from .core import BaseArranger
from .algorithm import AlgorithmMixin
from .renderer import RendererMixin

class SeatArranger(BaseArranger, AlgorithmMixin, RendererMixin):
    """
    學生座位排列核心邏輯模組 (已重構為多模組架構)
    透過 Mixin 模式繼承演算法與渲染功能，保持主類別乾淨。
    """
    def handle_request(self, data):
        """統一處理前端的各種請求"""
        action = data.get('action', 'arrange')
        custom_cls = data.get('customClassrooms', [])
        all_classrooms = self.CLASSROOMS + custom_cls
        
        if action == 'get_config':
            for c in all_classrooms:
                c['capacity'] = self.get_capacity(c)
            return {"classrooms": all_classrooms}
            
        elif action == 'arrange':
            return self.process_arrangement(data, all_classrooms)
            
        elif action == 'render_only':
            return self.generate_html(
                data.get('selectedCls', []), 
                all_classrooms, 
                data.get('seatMap', {}), 
                data.get('students', []), 
                data.get('quincunx', False)
            )
