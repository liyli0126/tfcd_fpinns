# PINN 框架架构文档

## 整体架构

```
main/
├── run_tfcd_pinn.m          # 主程序入口
params_data/
├── initialize_parameters.m   # 参数初始化
├── generate_training_data.m  # 训练数据生成和更新
training_core/
├── create_network.m         # 网络创建
├── train_two_stage_pinn.m   # 两阶段训练主控
├── train_adam.m             # Adam优化阶段
├── train_lbfgs.m            # L-BFGS优化阶段
├── loss_fractional_pinn.m   # 损失函数计算
utils/
├── weight_functions.m       # 权重计算系统
├── numerical_constants.m    # 数值常量管理
├── rar_sampling.m          # RAR采样策略
├── sample_iteration_data.m  # 迭代数据采样

├── MFL1_Caputo.m           # Caputo分数阶导数
visualization/
├── plot_*.m                # 可视化函数
```

## 核心接口设计

### 1. 主程序接口 (`run_tfcd_pinn.m`)

**功能**: 程序入口点，协调所有模块
**接口**: 无参数，无返回值
**流程**: 
1. 路径设置
2. 参数初始化
3. 网络创建
4. 训练执行
5. 结果可视化

### 2. 参数管理接口

#### `initialize_parameters()`
**功能**: 初始化所有训练参数
**输入**: 无
**输出**: `params` - 参数结构体
**包含**:
- 问题参数 (α, D, v, L, T)
- 训练参数 (epochs, learning rates)
- 权重参数 (λ_pde, λ_bc, λ_ic)
- 采样参数 (RAR频率, 批次大小)

#### `generate_training_data(params, net, iter)`
**功能**: 生成和更新训练数据
**输入**: 
- `params` - 参数结构体
- `net` - 神经网络 (可选，用于RAR采样)
- `iter` - 当前迭代次数 (可选，用于RAR采样)
**输出**: `params` - 更新后的参数结构体
**功能**:
- 初始数据生成 (无net/iter参数时)
- 动态数据更新 (训练过程中，包含RAR采样)
- 生成历史点、边界点、初始条件点

### 3. 网络训练接口

#### `train_two_stage_pinn(net, params)`
**功能**: 两阶段训练主控
**输入**: 
- `net` - 神经网络
- `params` - 训练参数
**输出**: 
- `net` - 训练后的网络
- `loss_history` - 损失历史
**阶段**:
1. Adam阶段 (全局探索)
2. L-BFGS阶段 (精细调优)

#### `train_adam(net, params, maxIter)`
**功能**: Adam优化器训练
**输入**: 
- `net` - 神经网络
- `params` - 训练参数
- `maxIter` - 最大迭代次数
**输出**: 
- `net` - 更新后的网络
- `loss_history` - 损失历史

#### `train_lbfgs(net, params, maxIter)`
**功能**: L-BFGS优化器训练
**输入**: 
- `net` - 神经网络
- `params` - 训练参数
- `maxIter` - 最大迭代次数
**输出**: 
- `net` - 更新后的网络
- `loss_history` - 损失历史

### 4. 数据采样接口

#### `sample_iteration_data(params, iter, net)`
**功能**: 为每次迭代采样训练数据
**输入**: 
- `params` - 参数结构体
- `iter` - 当前迭代次数
- `net` - 神经网络 (用于RAR)
**输出**: 所有训练数据点 (dlarray格式)
**策略**:
- 固定边界和初始条件点
- 动态生成配点 (随机或RAR)

#### `rar_sampling(params, net)`
**功能**: 残差自适应细化采样
**输入**: 
- `params` - 参数结构体
- `net` - 神经网络
**输出**: 
- `t_r, x_r` - 选中的配点
**采样策略**:
1. 时间分层采样
2. 边界区域采样
3. 关键时间区域采样
4. 峰值区域采样

### 5. 损失计算接口



#### `loss_fractional_pinn(...)`
**功能**: 分数阶PINN损失函数
**包含**:
- PDE残差损失
- 边界条件损失
- 初始条件损失
- Sobolev正则化
- 峰值保护损失

### 6. 权重系统接口

#### `weight_functions(type, params, varargin)`
**功能**: 统一的权重计算系统
**输入**: 
- `type` - 权重类型
- `params` - 参数结构体
- `varargin` - 额外参数
**输出**: `weights` - 计算得到的权重
**支持类型**:
- `peak_region` - 峰值区域权重
- `boundary_region` - 边界区域权重
- `time_weighting` - 时间权重
- `local_weighting` - 局部权重
- `rar_selection` - RAR选择权重
- `ic_weighting` - 初始条件权重
- `longterm_weighting` - 长期预测权重

## 数据流设计

### 训练数据流
```
generate_training_data() → sample_iteration_data() → loss_fractional_pinn() → optimizer
```

### 权重计算流
```
weight_functions() → loss_fractional_pinn()
```

### RAR采样流
```
rar_sampling() → sample_iteration_data() → loss_fractional_pinn()
```

## 接口设计原则

### 1. 单一职责
- 每个函数只负责一个明确的功能
- 避免功能重叠和重复实现

### 2. 清晰命名
- 函数名反映功能
- 变量名具有自解释性
- 避免缩写和模糊命名

### 3. 统一接口
- 相似功能使用一致的调用方式
- 参数顺序和类型保持一致
- 返回值格式标准化

### 4. 模块化设计
- 功能模块独立，依赖关系清晰
- 便于测试和维护
- 支持功能扩展

### 5. 错误处理
- 输入参数验证
- 异常情况处理
- 友好的错误信息

## 扩展性设计

### 1. 新增权重类型
在 `weight_functions.m` 中添加新的 case 分支

### 2. 新增采样策略
在 `rar_sampling.m` 中添加新的采样方法

### 3. 新增损失项
在 `loss_fractional_pinn.m` 中添加新的损失组件

### 4. 新增优化器
在 `training_core/` 中添加新的训练函数

## 性能优化

### 1. 内存管理
- 使用 dlarray 避免数据转换
- 及时释放临时变量

### 2. 计算优化
- 批量计算减少循环
- 向量化操作提高效率

### 3. 采样优化
- RAR采样减少无效点
- 分层采样提高覆盖率
