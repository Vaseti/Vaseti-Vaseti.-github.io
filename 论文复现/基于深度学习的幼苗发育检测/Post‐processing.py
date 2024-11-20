import numpy as np

# 假设 predictions 是一个包含模型预测标签的列表，包含多帧时间步的预测（例如4帧对应一个小时的观察）
# predictions 是一个 形状为 [n_frames, n_samples] 的数组，其中 n_frames 是时间步数量，n_samples 是每个时间步的预测标签数

def post_processing_smoothing(predictions, window_size=4):
    """
    使用滑动窗口对预测标签进行平滑处理，避免因细微波动引起的错误标签预测。
    
    参数:
    - predictions: 形状为 [n_frames, n_samples] 的二维数组，每行是一个时间步的预测标签
    - window_size: 滑动窗口的大小，默认为4（即一个小时的观察）

    返回:
    - smoothed_predictions: 经平滑处理后的预测结果
    """
    n_frames, n_samples = predictions.shape  # 获取时间步和样本数
    smoothed_predictions = np.copy(predictions)  # 初始化平滑后的预测，初始为原始预测

    # 滑动窗口平滑
    for t in range(window_size, n_frames - window_size):  # 从第window_size到倒数第window_size的时间步进行处理
        # 获取当前窗口内的预测标签
        window = predictions[t - window_size : t + window_size + 1]  # 滑动窗口范围
        # 对窗口内的每个样本进行中位数处理，防止阶段变化过于剧烈
        smoothed_predictions[t] = np.median(window, axis=0)

    # 最后返回平滑后的预测结果
    return smoothed_predictions

# 假设模型的预测结果（多帧标签数据）
predictions = np.random.randint(0, 4, size=(10, 5))  # 这里生成一个假设的预测结果，10帧，5个样本

# 进行后处理
smoothed_predictions = post_processing_smoothing(predictions, window_size=4)

# 打印处理前后的预测结果
print("Original Predictions:")
print(predictions)
print("Smoothed Predictions:")
print(smoothed_predictions)
