# 实验改动记录

## 2026-08-14：Glossy 引导的镜面渲染监督

- 使用停止梯度的 glossy 置信度图，引导最终渲染监督。
- 新增 glossy 区域加权的 RGB loss 和 Haar 高频 loss。
- 将辅助 loss 的梯度主要传递给 attenuation 和镜面渲染分支。
- 将 glossy 置信度加入原有的 transmission 梯度路由，减少镜面内容被透射分支解释。
- 在 `tandt-eval.sh` 中启用较保守的 Truck 实验参数。
- 新增 `glossy_guidance_gate` 和 `final_specular` 渲染诊断图。
- 本次消融实验关闭外部法线先验监督，即 `lambda_normal_prior=0`。
