# convert_to_torchscript.py

import torch
# 从您的训练脚本中导入模型类的定义
from train_oximeter import OximeterCNN


def convert_model():
    """
    这个函数会加载您训练好的 .pth 文件,
    并将其转换为移动端可以使用的 TorchScript (.ptl) 格式。
    """
    # --- 配置区域 ---
    # 您的PyTorch模型权重文件路径
    pytorch_model_path = './oximeter_model.pth'
    # 转换后输出的TorchScript Lite模型文件路径
    torchscript_lite_path = './oximeter_model.ptl'
    # ----------------

    print(f"正在加载PyTorch模型: {pytorch_model_path}")

    # 1. 初始化与训练时完全相同的模型架构
    model = OximeterCNN()

    # 2. 加载您训练好的权重
    #    注意：如果您在Windows上训练，在Mac/Linux上加载，或者反之，
    #    请使用 map_location=torch.device('cpu') 来避免设备不匹配错误。
    model.load_state_dict(torch.load(pytorch_model_path, map_location=torch.device('cpu')))

    # 3. 将模型设置为评估模式（这很重要，会禁用Dropout等训练专用层）
    model.eval()

    print("模型加载成功，正在转换为TorchScript...")

    # 4. 使用JIT跟踪（tracing）来创建TorchScript模型
    #    我们需要一个符合模型输入尺寸的示例输入
    example_input = torch.rand(1, 3, 90)  # (批大小, RGB通道, 帧数)
    traced_script_module = torch.jit.trace(model, example_input)

    print("转换为TorchScript成功，正在优化为移动端格式...")

    # 5. (推荐) 将TorchScript模型优化为移动端专用的Lite格式
    from torch.utils.mobile_optimizer import optimize_for_mobile
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)

    # 6. 保存优化后的Lite模型
    traced_script_module_optimized._save_for_lite_interpreter(torchscript_lite_path)

    print("-" * 50)
    print(f"转换成功! 🚀")
    print(f"您的移动端模型已保存至: {torchscript_lite_path}")
    print("现在，请将这个 .ptl 文件添加到您的安卓App的 'assets' 文件夹中。")
    print("-" * 50)


if __name__ == '__main__':
    convert_model()