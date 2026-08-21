"""Pre-build the optional 3DGRT extension before a long training run."""

from .three_dgrt import load_3dgrt_extension


if __name__ == '__main__':
    load_3dgrt_extension(verbose=True)
    print('3DGRT CUDA/OptiX extension is ready.')
