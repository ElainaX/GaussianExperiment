
# 叠加在渲染图上，加轻微模糊
python render_gaussian_heatmap.py -m ~/autodl-tmp/model/rtsplat/tandt/fastgs_densify_off+remove_lpips/truck --overlay

# 更激进的 gamma 拉伸
python render_gaussian_heatmap.py -m ~/autodl-tmp/model/rtsplat/tandt/fastgs_densify_off+remove_lpips/truck --scale gamma --gamma 0.25