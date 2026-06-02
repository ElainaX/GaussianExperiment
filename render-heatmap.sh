# 默认：test 相机，uniform 计数
# python render_gaussian_heatmap.py -m ~/autodl-tmp/model/rtsplat/tandt/fastgs_off+remove_lpips/truck

# 叠加在渲染图上，加轻微模糊
python render_gaussian_heatmap.py -m ~/autodl-tmp/model/rtsplat/tandt/fastgs_off+remove_lpips/truck --overlay --sigma 1.5

# 训练集相机，按 opacity×occupancy 加权
# python render_gaussian_heatmap.py -m ~/autodl-tmp/model/rtsplat/tandt/fastgs_off+remove_lpips/truck --cameras train --weight opacity

# 只看第 0、4、9 张 test 相机，指定 checkpoint
# python render_gaussian_heatmap.py -m ~/autodl-tmp/model/rtsplat/tandt/fastgs_off+remove_lpips/truck --iteration 30000 --cameras 0 4 9

# 交互显示（需要 TkAgg 后端）
# python render_gaussian_heatmap.py -m ~/autodl-tmp/model/rtsplat/tandt/fastgs_off+remove_lpips/truck --show
# 结果保存在 <model_path>/heatmap/<iter>/ 下。--overlay 时输出左右两列（叠加图 + 纯热力图）。scipy 是唯一额外依赖（用于 --sigma 模糊），不传 --sigma 时不需要。