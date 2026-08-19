# 实验改动记录

## 实验提交与输出目录约定

- 后续每次代码修改完成并验证后都提交到 Git，不把不同实验的改动混在同一个提交中。
- 提交信息用中文概括关键改动；交付时同时说明 commit hash 和主要内容。
- `tandt-eval.sh` 中的 `EXPERIMENT_NAME` 使用一个简短英文单词概括实验目的。
- 输出目录由脚本根据当前 `HEAD` 自动生成，格式为 `<实验名>_<提交日期MMDD>_<hash前六位>/truck`，例如 `probe_0819_a1b2c3/truck`。
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

## 2026-08-16：固定辐射 Reflection Probe

- 废弃自由学习的 signed cubemap residual；它在车窗上退化成了类似薄膜干涉的彩色误差纹理。
- 将 Probe 限制在环境中心半径 15 的局部范围，避免最远点采样把探针散到远景。
- 对高 glossy 高斯做加权空间聚类，并将捕获位置朝最近训练相机外移 0.25。
- 捕获目标 Probe 时排除该 glossy 区域自身，复用现有渲染器生成六个 90°、64×64 的固定 cubemap。
- cubemap 保存非负场景辐射和 alpha 有效性；未命中几何的方向继续使用全局 SphMip。
- 最终镜面改为全局环境与局部 Probe 的能量混合，不再叠加可正可负的学习残差。
- Probe 在第 30000 次迭代捕获一次，之后保持固定，不接收 RGB loss 梯度。
- 捕获的六面图和有效性 mask 输出到 `probe_cubemaps/iteration_30000`，便于直接检查是否包含树木/建筑。
- 新增 `local_probe_radiance` 调试图，实验目录改为 `test_priors_captured_light_probe`。

## 2026-08-18：SOTA 基线上的 128×128 Reflection Probe 消融

- 严格恢复 2026-08-11 SOTA 提交 `8b13988` 的镜面机制：`glossy_specular_boost=1.0`。
- 恢复高 glossy 高斯的材质边界：`roughness≤0.15`、`reflectance≥0.70`。
- Probe 在 SOTA 的全局 SphMip 镜面结果上混合；关闭 Probe 时退化回原 SOTA 渲染公式。
- 固定 cubemap 从每面 64×64 提高到 128×128，便于检查捕获内容、孔洞和视角错位。
- 新增推理参数 `--disable_local_probe`，同一 checkpoint 可以直接做 Probe ON/OFF 消融。
- 新增 `--render_tag`，分别输出 `ours_61000_probe_on` 和 `ours_61000_probe_off`，防止互相覆盖。
- `tandt-eval.sh` 使用 `env_scope_radius=1`，新实验目录为 `test_priors_sota_captured_probe_128`。

## 2026-08-18：恢复工作树并固定 SOTA 法线默认值

- 中止误发起的 `git revert 8b13988`，清除 `train.py` 与 `tandt-eval.sh` 的提交冲突，恢复到提交 `3589adc`。
- 明确将 `lambda_normal_prior=0` 作为默认 SOTA 设置；法线先验只输出诊断图，不参与训练 loss。
- `tandt-eval.sh` 继续显式传入 `--lambda_normal_prior 0`，需要法线监督时必须手动传入正权重。
