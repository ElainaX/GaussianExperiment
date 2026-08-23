# 实验改动记录

## 实验提交与输出目录约定

- 后续每次代码修改完成并验证后都提交到 Git，不把不同实验的改动混在同一个提交中。
- 提交信息用中文概括关键改动；交付时同时说明 commit hash 和主要内容。
- `tandt-eval.sh` 中的 `EXPERIMENT_NAME` 使用一个简短英文单词概括实验目的。
- 输出目录由脚本根据当前 `HEAD` 自动生成，格式为 `<实验名>_<提交日期MMDD>_<hash前六位>/truck`，例如 `raytrace_0821_a1b2c3/truck`。
- 若存在已跟踪但未提交的修改，脚本会给出警告，避免把实验结果错误归属到当前 commit。

## 2026-08-14：Glossy 引导的镜面渲染监督

- 使用停止梯度的 glossy 置信度图，引导最终渲染监督。
- 新增 glossy 区域加权的 RGB loss 和 Haar 高频 loss。
- 将辅助 loss 的梯度主要传递给 attenuation 和镜面渲染分支。
- 将 glossy 置信度加入原有的 transmission 梯度路由，减少镜面内容被透射分支解释。
- 在 `tandt-eval.sh` 中启用较保守的 Truck 实验参数。
- 新增 `glossy_guidance_gate` 和 `final_specular` 渲染诊断图。
- 本次消融实验关闭外部法线先验监督，即 `lambda_normal_prior=0`。

## 2026-08-16：第二阶段镜面精修

- 将 glossy RGB/Haar 监督推迟到第 30000 次迭代，并使用 5000 次迭代预热。
- 进入镜面精修阶段后，冻结几何、SH、occupancy、opacity 和 transmissivity。
- 精修阶段只训练 SphMip、Light MLP、roughness、reflectance 和局部材质特征。
- glossy 区域同时冻结 scatter 与 transmission 的基础外观梯度，但保留 attenuation 梯度。
- 第二阶段将主 PBR loss 和 edge loss 的有效梯度限制在 glossy gate 内，避免镜面网络拟合整辆车的残差。
- 关闭 roughness/reflectance 硬性覆盖，改用权重为 0.002 的单边软材质约束。
- 将软材质目标放宽为 `roughness≤0.25、reflectance≥0.60`。
- 将 `glossy_specular_boost` 设为 0，避免错误镜面结果被再次放大。
- 训练日志改用 `high-score`，表示超过阈值但不再代表被强制修改材质。
- 增加配置检查，禁止在增殖结束前冻结高斯，且精修必须同时启用 glossy prior 与梯度路由。
- 使用新的实验目录 `test_priors_glossy_refine_stage2`。

## 2026-08-18：恢复工作树并固定 SOTA 法线默认值

- 中止误发起的 `git revert 8b13988`，清除 `train.py` 与 `tandt-eval.sh` 的提交冲突，恢复到提交 `3589adc`。
- 明确将 `lambda_normal_prior=0` 作为默认 SOTA 设置；法线先验只输出诊断图，不参与训练 loss。
- `tandt-eval.sh` 继续显式传入 `--lambda_normal_prior 0`，需要法线监督时必须手动传入正权重。

## 2026-08-21：高 Glossy 区域的 3DGRT 二次反射

- 删除旧的局部环境贴图实现、参数、checkpoint、捕获流程和诊断图。
- 固定 NVIDIA 官方 `3dgrut` 源码版本，新增独立 CUDA/OptiX 二次光线模块。
- 主视角继续使用原来的 `diff-surfel-anych`；只有高 glossy、低 roughness 像素发射反射射线。
- 使用反射表面位置加偏移作为射线起点，3DGRT 命中辐射按 glossy gate 和命中 opacity 与原 SphMip 镜面结果混合。
- 2D glossy prior 加入 roughness 硬门控，先验 roughness 大于等于阈值时分数固定为零，不参与多视角高斯累积。
- 新增二次射线辐射、混合 gate、命中 opacity 和 roughness gate 诊断图。
- `tandt-eval.sh` 实验名改为 `raytrace`，并在训练前预编译 3DGRT，尽早暴露 CUDA/OptiX 环境问题。

## 2026-08-21：AutoDL 强制同步 3DGRT 子模块

- `rungit.sh` 默认同步 `origin/rtsplat-test`，避免误回退到不含 3DGRT 的 baseline 分支。
- 主仓库更新后强制同步并检出固定版本的官方 `3dgrut` 子模块。
- 单独初始化 3DGRT 编译必需的 OptiX headers，不额外下载当前未使用的依赖。
- 支持通过 `bash rungit.sh <分支名>` 临时选择其他远端分支。

## 2026-08-21：Glossy 区域法线变化率约束

- 保持原有逐像素/区域平均法线先验监督关闭，即 `lambda_normal_prior=0`。
- 新增仅作用于高 glossy、低先验 roughness 像素的相邻法线变化率 loss。
- 使用 `1-dot(n_p,n_q)` 比较相邻法线夹角，不依赖先验与渲染法线的全局坐标轴约定。
- 采用单边惩罚：允许渲染表面比先验更平，只抑制超过先验与 margin 的额外弯曲。
- 使用先验深度连续性阻断跨物体、前后景边界的错误平滑。
- depth、normal、roughness 先验均按需缓存为 CPU `uint8`，避免逐迭代重复解码 PNG。
- loss 从增殖结束的第 15000 次迭代开始，经 5000 次迭代预热，默认实验权重为 0.02。
- 新增渲染/先验法线变化率、超额弯曲误差和实际约束 mask 的 `vis` 输出。
- `tandt-eval.sh` 实验名改为 `normrate`。

## 2026-08-22：3DGRT 多视角高斯可靠性诊断

- 增殖结束后，用训练相机下 alpha/transmittance 加权的有效像素覆盖计算每个高斯的多视角可靠性 `R`。
- `R` 同时考虑有效可见视角数和每个有效视角的平均像素覆盖；默认分别在 8 个视角和 16 个像素处饱和。
- 在 15000～30000 次迭代之间每 5000 次迭代采样 32 个训练相机，并以 0.5 EMA 更新可靠性。
- 将 `raytrace_reliability` 保存到 PLY；旧 PLY 缺少该字段时安全回退为零。
- 额外用方向无关 SH 对二次射线命中的高斯可靠性做 alpha 加权积分，输出 `secondary_raytrace_reliability`。
- 可靠性追踪只在 `render.py` 显式开启 debug 时执行，避免训练阶段把 3DGRT 工作量翻倍。
- 本阶段只做诊断，不使用 `R` 过滤 BVH、修改 hit opacity 或改变 SphMip/3DGRT 混合结果。
- `tandt-eval.sh` 实验名改为 `reliability`。

## 2026-08-22：修正 Window PSNR 的统计区域

- 训练日志中的 Window PSNR 改为只在白色透明区域 mask 内计算 RGB MSE，mask 外像素不再以黑色背景参与分母。
- 同步修正 Opaque PSNR，只统计非透明区域，且平均时跳过不存在对应区域的相机。
- `metrics.py` 输出的 `Masked PSNR` 使用同一套有效像素统计，保留原 JSON 字段名以兼容已有分析脚本。
- Masked SSIM 与 Masked LPIPS 的历史定义暂时保持不变；本次只修正用户指定的 PSNR。
- `tandt-eval.sh` 实验名改为 `maskpsnr`。

## 2026-08-23：3DGRT 不可靠命中 Heatmap

- 新增不可靠命中分数 `U = hit_opacity × (1 - reliability)`，只有实际命中且命中高斯可靠性低时才产生高响应。
- 开启 reliability debug 时，`vis` 同时导出灰度分数 `secondary_raytrace_unreliable_hit_*` 和 Turbo 彩色图 `secondary_raytrace_unreliable_hit_heatmap_*`。
- 彩色图中未发射或未命中的像素强制为黑色；命中越不可靠，颜色由蓝色逐渐变为黄色、红色。
- 新增离线处理脚本，可直接组合已有 `vis` 中的 hit opacity 与 reliability PNG，无需重新训练或重新运行 3DGRT。
- 离线递归处理时忽略 `.ipynb_checkpoints` 等隐藏目录，避免为编辑器缓存重复生成结果。
- `tandt-eval.sh` 实验名改为 `unreliable`。

## 2026-08-23：可靠性驱动的 3DGRT/SphMip 路由

- 新增可关闭的 `secondary_raytrace_routing_on`，关闭时保持原 3DGRT 混合公式不变。
- 将二次射线命中的多视角可靠性通过 Smoothstep 映射为路由分数；默认 `R≤0.35` 完全使用 SphMip，`R≥0.65` 使用 3DGRT，中间连续过渡。
- 最终 3DGRT 权重为 glossy 材质权重乘路由分数；低可靠命中和未命中均回退到原环境贴图。
- 路由分数停止梯度，不允许训练通过改变几何来操纵路由；可靠性追踪也不保留第二套 OptiX backward graph。
- 新增 `secondary_raytrace_route_score_*` 灰度图和 `secondary_raytrace_route_heatmap_*` 彩色可视化。
- 离线图片工具也可用相同阈值从已有 hit/reliability PNG 生成预测路由图，无需重新渲染。
- 启用路由训练时强制要求多视角可靠性计算，避免所有高斯因缺少 `R` 而错误回退环境贴图。
- `tandt-eval.sh` 实验名改为 `routing`。
