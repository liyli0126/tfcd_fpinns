# 权重函数系统说明

## 概述

`weight_functions.m` 是一个统一的权重计算管理系统，用于消除代码中的重复权重计算逻辑，提高代码的可维护性和一致性。

## 主要功能

### 1. 峰值区域权重 (`peak_region`)
- **用途**: 为解的最大值区域 (x ≈ 0.5) 分配更高权重
- **调用**: `weights = weight_functions('peak_region', params, x_coords);`
- **权重分配**:
  - 峰值区域 [0.4L, 0.6L]: 权重 8.0
  - 近峰值区域 [0.3L, 0.4L] ∪ [0.6L, 0.7L]: 权重 4.0
  - 前后区域 [0, 0.2L] ∪ [0.8L, L]: 权重 5.0

### 2. 边界区域权重 (`boundary_region`)
- **用途**: 为边界区域分配时间分层的权重
- **调用**: `weights = weight_functions('boundary_region', params, x_coords, t_coords);`
- **权重分配**:
  - 早期时间边界 (t < 0.3T): 权重 6.0
  - 中期时间边界 (0.3T ≤ t ≤ 0.7T): 权重 3.0
  - 晚期时间边界 (t > 0.7T): 权重 2.0

### 3. 时间权重 (`time_weighting`)
- **用途**: 为不同时间点分配权重，强调早期和晚期时间
- **调用**: `weights = weight_functions('time_weighting', params, t_coords);`
- **权重分配**:
  - 基础权重: 1 + 8(t/T)
  - 早期时间 (t < 0.3T): 额外 ×1.5
  - 晚期时间 (t > 0.7T): 额外 ×1.3

### 4. 局部权重 (`local_weighting`)
- **用途**: 结合空间位置和残差大小的综合权重
- **调用**: `weights = weight_functions('local_weighting', params, x_coords, residual_values);`
- **权重分配**: 空间权重 × 残差权重

### 5. RAR选择权重 (`rar_selection`)
- **用途**: 为RAR采样策略分配选择权重
- **调用**: `weights = weight_functions('rar_selection', params, x_coords, t_coords);`
- **权重分配**:
  - 边界区域: 权重 0.8 (减少选择)
  - 峰值区域: 权重 1.3 (增加选择)
  - 内部区域: 权重 1.0 (标准)

### 6. 初始条件权重 (`ic_weighting`)
- **用途**: 为初始条件损失分配权重，强调峰值区域
- **调用**: `weights = weight_functions('ic_weighting', params, x_coords);`
- **权重分配**:
  - 峰值区域: 权重 3.0
  - 近峰值区域: 权重 2.0
  - 其他区域: 权重 1.0

### 7. 长期预测权重 (`longterm_weighting`)
- **用途**: 为长期预测 (t > 0.7T) 分配更高权重
- **调用**: `weights = weight_functions('longterm_weighting', params, t_coords);`
- **权重分配**:
  - 长期区域 (t > 0.7T): 权重 3.0
  - 其他区域: 权重 1.0

## 数值常量管理

`numerical_constants.m` 统一管理所有数值稳定性参数：

- `epsilon`: 数值稳定性参数 (1e-12)
- `epsilon_loss`: 损失函数稳定性参数 (1e-8)
- `epsilon_grad`: 梯度稳定性参数 (1e-10)
- `perturbation_factor`: 扰动因子 (0.01)
- `weight_decay`: 权重衰减 (0.95)
- 各种阈值参数 (时间、空间、边界等)

## 使用示例

### 在损失函数中使用
```matlab
% 计算局部权重
local_weight = weight_functions('local_weighting', params, x_r, residual);

% 计算时间权重
t_weight = weight_functions('time_weighting', params, t_r);

% 计算初始条件权重
ic_weights = weight_functions('ic_weighting', params, x_ic);
```

### 在RAR采样中使用
```matlab
% 计算RAR选择权重
selection_weights = weight_functions('rar_selection', params, x_rar_cand, t_rar_cand);
```

## 优势

1. **消除重复代码**: 统一的权重计算逻辑
2. **提高可维护性**: 权重策略集中管理
3. **确保一致性**: 所有模块使用相同的权重策略
4. **易于调试**: 权重计算逻辑集中，便于问题定位
5. **参数化配置**: 通过数值常量文件统一管理参数

## 注意事项

1. 确保在调用权重函数前已正确设置 `params` 结构体
2. 权重函数返回的数组需要转换为 `dlarray` 类型用于深度学习计算
3. 数值常量文件应在权重函数之前加载
4. 新增权重类型时，需要在主函数中添加相应的 case 分支

