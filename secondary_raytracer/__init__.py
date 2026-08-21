"""Optional secondary-ray backends for RT-Splatting."""

from .three_dgrt import ThreeDGRTSecondaryTracer, load_3dgrt_extension

__all__ = ['ThreeDGRTSecondaryTracer', 'load_3dgrt_extension']
