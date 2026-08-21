# 3DGRT 二次反射模块

本目录只负责高 glossy 像素的二次反射射线。主视角仍由
`diff-surfel-anych` 光栅化，FastGS 的增殖、剪枝和梯度统计不变。

CUDA/OptiX 内核来自固定版本的 NVIDIA 官方 `3dgrut` 子模块。首次使用：

```bash
git submodule update --init submodules/3dgrut
git -C submodules/3dgrut submodule update --init threedgrt_tracer/dependencies/optix-dev
python -m secondary_raytracer.build_3dgrt
```

需要 CUDA Toolkit、Ninja 和支持 OptiX 的 NVIDIA RTX GPU。也可以用环境变量
`RTSPLAT_3DGRT_ROOT` 指向另一个完整的官方 `3dgrut` 源码目录。

训练时不传 `--secondary_raytrace_on` 就完全关闭该模块；已有 checkpoint
渲染消融时可传 `--disable_secondary_raytrace`，不会创建 tracer 或构建 BVH。
