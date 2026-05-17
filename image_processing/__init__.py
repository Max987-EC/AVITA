# 影像處理模組彙整 (Image Processing Module Aggregation)
# 此檔案負責將各個功能模組 (Mixins) 整合進主處理器 ImageProcessor。

from .core import BaseProcessor
from .intensity import IntensityMixin
from .spatial import SpatialMixin
from .frequency import FrequencyMixin
from .feature import FeatureMixin
from .morphology import MorphologyMixin
from .analysis import AnalysisMixin       # 分析與測量模組 (Connected Components, Features)
from .segmentation import SegmentationMixin # 影像分割模組 (Region Growing, Split-Merge)

class ImageProcessor(
    BaseProcessor, 
    IntensityMixin, 
    SpatialMixin, 
    FrequencyMixin, 
    FeatureMixin, 
    MorphologyMixin, 
    AnalysisMixin, 
    SegmentationMixin
):
    """
    綜合影像處理器 (Comprehensive Image Processor):
    透過多重繼承整合了以下功能：
    - BaseProcessor: 基礎設定、直方圖與頻譜產生
    - IntensityMixin: 強度轉換與二值化 (線性、對數、Gamma、直方圖等化)
    - SpatialMixin: 空間域濾波 (平滑、銳化、邊緣偵測)
    - FrequencyMixin: 頻率域濾波 (低通、高通、帶阻、Notch 濾波)
    - FeatureMixin: 特徵檢測 (霍夫直線與圓形偵測)
    - MorphologyMixin: 形態學運算 (侵蝕、膨脹、開閉、Hit-or-Miss)
    - AnalysisMixin: 連通區域分析與進階特徵量測 (次像素精度)
    - SegmentationMixin: 影像分割演算法 (區域成長、區域分裂與合併)
    """
    pass
