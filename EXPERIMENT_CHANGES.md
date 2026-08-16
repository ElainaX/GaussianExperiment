# 实验改动记录

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

## 2026-08-16：局部 Light Probe 二次反射补偿

- 以 2026-08-11 的 SOTA 提交 `8b13988`（`add normal loss`）为代码基线。
- 回退 glossy RGB/Haar/材质专项监督、梯度路由和第二阶段参数冻结。
- glossy score 只保留为镜面区域标记，不再直接覆盖 roughness 或 reflectance。
- 增殖结束后，对高分高斯做空间最远点划分，初始化 8 个离散局部探针。
- 每个探针使用一个可学习的 32×32 六面 cubemap 残差。
- 高 glossy score、低 roughness 的像素按空间位置选择最近探针，并按反射方向查询补偿颜色。
- 局部探针由原始最终 RGB 重建 loss 训练，另加很小的 L2 正则，不新增 glossy 专项监督。
- 新增 `local_probe_gate`、`local_probe_index` 和 `local_probe_correction` 调试图。
- 使用新的实验目录 `test_priors_local_light_probe`。
