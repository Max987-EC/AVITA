# 負責將所有模組組裝起來

from .core import BaseProcessor
from .intensity import IntensityMixin
from .spatial import SpatialMixin
from .frequency import FrequencyMixin
from .feature import FeatureMixin
from .morphology import MorphologyMixin
from .analysis import AnalysisMixin       # 🌟 新增：分析與測量模組
from .segmentation import SegmentationMixin # 🌟 新增：進階分割模組

# 🌟 將 AnalysisMixin 與 SegmentationMixin 加入繼承列表
class ImageProcessor(BaseProcessor, IntensityMixin, SpatialMixin, FrequencyMixin, FeatureMixin, MorphologyMixin, AnalysisMixin, SegmentationMixin):
    """
    綜合影像處理器：
    繼承了基礎功能、強度轉換、空間濾波、頻率濾波、特徵檢測、形態學運算，
    以及新增的連通區域分析與區域成長分割。
    """
    pass
