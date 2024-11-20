import tensorflow as tf
from tensorflow.keras import layers, models

# 定义CNN-LSTM模型
def build_cnn_lstm_model(input_shape=(64, 64, 3), lstm_units=128, num_classes=4):
    """
    构建CNN-LSTM模型
    input_shape: 输入图像的形状 (高度, 宽度, 通道数)
    lstm_units: LSTM层中的单元数
    num_classes: 分类的类别数 (例如，4个类别: Soil, FA, OC, FL)
    """
    model = models.Sequential()  # 初始化顺序模型

    # Step 1: 构建卷积层部分 (CNN部分)
    # 第一层卷积层，64个3x3的滤波器，ReLU激活函数
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    # 最大池化层，池化区域为2x2
    model.add(layers.MaxPooling2D((2, 2)))

    # 第二层卷积层，128个3x3的滤波器，ReLU激活函数
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))  # 池化层

    # 第三层卷积层，256个3x3的滤波器
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # 第四层卷积层，256个3x3的滤波器
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))  # 池化层

    # Step 2: Flatten卷积层输出 (将二维特征图转换为一维向量)
    # 将CNN层的输出展平，以便传递给LSTM层
    model.add(layers.Flatten())

    # Step 3: 添加LSTM层 (LSTM部分)
    # 这里使用Reshape将CNN输出的特征调整为LSTM所需要的形状
    model.add(layers.Reshape((-1, 256)))  # 假设CNN部分输出的特征数是256，转化为LSTM可接收的形状
    model.add(layers.LSTM(lstm_units, activation='tanh', return_sequences=False))  # LSTM层，用于处理时序数据

    # Step 4: 全连接层和输出层
    model.add(layers.Dense(512, activation='relu'))  # 全连接层
    model.add(layers.Dropout(0.5))  # Dropout层，用于防止过拟合
    model.add(layers.Dense(num_classes, activation='softmax'))  # 输出层，使用softmax激活函数用于分类任务

    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# 模型的构建
model = build_cnn_lstm_model(input_shape=(64, 64, 3), lstm_units=128, num_classes=4)

# 打印模型架构
model.summary()  # 输出模型的详细结构
