# 負責將所有模組組裝起來

from .core import BaseProcessor
from .intensity import IntensityMixin
from .spatial import SpatialMixin
from .frequency import FrequencyMixin
from .feature import FeatureMixin

class ImageProcessor(BaseProcessor, IntensityMixin, SpatialMixin, FrequencyMixin, FeatureMixin):
    """
    綜合影像處理核心類別 (已重構為多模組架構)
    透過 Mixin 模式繼承各個子模組的功能，保持主類別乾淨。
    """
    pass
