import torch
data = torch.load('skeleton_prototypes.pth')

# 获取全局特征张量
global_prototype = data['global']  # type: torch.Tensor

# 验证张量属性
print(f"""
[张量验证]
  形状: {global_prototype.shape}
  数据类型: {global_prototype.dtype}
  设备位置: {global_prototype.device}
  数值范围: [{global_prototype.min():.4f}, {global_prototype.max():.4f}]
""")