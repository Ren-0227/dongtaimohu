import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
import graphviz
import os
from innovative_deblur import (
    InnovativeDeblurGAN,
    MotionAwareAttention,
    AdaptiveDeblurBlock,
    TemporalCoherenceModule
)

def create_network_visualization(model, input_shape=(1, 3, 128, 128), output_dir="network_visualization"):
    """
    创建网络架构可视化
    Args:
        model: 创新去模糊网络模型
        input_shape: 输入张量的形状
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(output_dir)
    
    # 创建示例输入
    dummy_input = torch.randn(input_shape)
    dummy_previous = torch.randn(input_shape)
    
    # 1. 可视化完整网络结构
    writer.add_graph(model, (dummy_input, dummy_previous))
    
    # 2. 可视化生成器结构
    writer.add_graph(model.generator, (dummy_input, torch.randn(1, 2, 128, 128), dummy_previous))
    
    # 3. 可视化运动估计器结构
    writer.add_graph(model.motion_estimator, (dummy_input, dummy_previous))
    
    # 4. 可视化判别器结构
    writer.add_graph(model.discriminator, dummy_input)
    
    # 添加网络说明
    writer.add_text('Network_Description', '''
    # 创新去模糊网络架构说明
    
    ## 1. 生成器 (InnovativeDeblurGenerator)
    - 初始特征提取: Conv3×3(3→64) + ReLU
    - 运动感知注意力: MotionAwareAttention(64)
    - 自适应去模糊块: 4×AdaptiveDeblurBlock(64)
    - 时序一致性模块: TemporalCoherenceModule(64)
    - 输出重建: Conv3×3(64→3) + Tanh
    
    ## 2. 运动估计器 (MotionEstimator)
    - 编码器: Conv3×3(6→64→128→256) + 下采样
    - 解码器: Deconv(256→128→64→2) + 双线性插值
    
    ## 3. 判别器
    - 多层卷积: Conv4×4 + LeakyReLU + BatchNorm
    - 输出层: Conv4×4 + Sigmoid
    ''')
    
    # 使用torchviz生成详细网络结构图
    try:
        # 生成完整网络结构图
        y, _ = model(dummy_input, dummy_previous)
        dot = make_dot(y, params=dict(model.named_parameters()))
        dot.render(f"{output_dir}/full_network", format="png")
        
        # 生成生成器结构图
        gen_output = model.generator(dummy_input, torch.randn(1, 2, 128, 128), dummy_previous)
        dot_gen = make_dot(gen_output, params=dict(model.generator.named_parameters()))
        dot_gen.render(f"{output_dir}/generator", format="png")
        
        # 生成运动估计器结构图
        motion_map = model.motion_estimator(dummy_input, dummy_previous)
        dot_motion = make_dot(motion_map, params=dict(model.motion_estimator.named_parameters()))
        dot_motion.render(f"{output_dir}/motion_estimator", format="png")
        
    except Exception as e:
        print(f"生成详细网络结构图时出错: {str(e)}")
    
    # 关闭写入器
    writer.close()
    
    print(f"网络架构可视化已保存到 {output_dir} 目录")
    print("使用以下命令查看TensorBoard可视化结果：")
    print(f"tensorboard --logdir={output_dir}")

def visualize_module_connections():
    """
    可视化模块间的连接关系
    """
    # 创建有向图
    dot = graphviz.Digraph(comment='Network Module Connections')
    dot.attr(rankdir='TB')
    
    # 添加节点
    dot.node('input', '输入图像')
    dot.node('motion_est', '运动估计器')
    dot.node('gen', '生成器')
    dot.node('disc', '判别器')
    dot.node('output', '去模糊图像')
    
    # 添加子模块节点
    with dot.subgraph(name='cluster_generator') as c:
        c.attr(label='生成器组件')
        c.node('init', '初始特征提取')
        c.node('motion_att', '运动感知注意力')
        c.node('deblur', '自适应去模糊块')
        c.node('temporal', '时序一致性模块')
        c.node('output_layer', '输出重建')
    
    # 添加连接
    dot.edge('input', 'motion_est')
    dot.edge('input', 'gen')
    dot.edge('motion_est', 'gen')
    dot.edge('gen', 'disc')
    dot.edge('gen', 'output')
    
    # 生成器内部连接
    dot.edge('init', 'motion_att')
    dot.edge('motion_att', 'deblur')
    dot.edge('deblur', 'temporal')
    dot.edge('temporal', 'output_layer')
    
    # 保存图
    dot.render('network_visualization/module_connections', format='png', cleanup=True)

def main():
    # 创建模型实例
    model = InnovativeDeblurGAN()
    
    # 创建网络架构可视化
    create_network_visualization(model)
    
    # 创建模块连接关系可视化
    visualize_module_connections()
    
    print("网络架构可视化完成！")
    print("1. 使用TensorBoard查看详细网络结构")
    print("2. 查看network_visualization目录下的PNG文件")
    print("3. 查看module_connections.png了解模块连接关系")

if __name__ == '__main__':
    main()