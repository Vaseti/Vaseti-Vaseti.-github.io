import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Conv3D, BatchNormalization, Activation

# 初始化一个顺序模型
model = Sequential()

# ConvLSTM2D层：
# - filters: 64表示卷积核的数量，即输出空间的维度
# - kernel_size: (3, 3)表示卷积核的尺寸为3x3
# - padding='same'表示输出与输入的尺寸相同，通过对输入的边缘进行填充
# - return_sequences=True意味着输出不仅是最后一个时间步的隐藏状态，而是每个时间步的隐藏状态
# - input_shape=(None, 64, 64, 1)表示输入数据的形状，其中None代表时间步数是动态的，64x64是图像的高和宽，1表示单通道（灰度图）
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', 
                     return_sequences=True, input_shape=(None, 64, 64, 1)))

# BatchNormalization层：
# - 这层会规范化前一层的输出，使得输出更有利于后续的训练过程
# - 通过标准化每一层的输入，提高模型的训练速度，并减少参数更新过程中的不稳定性
model.add(BatchNormalization())

# 激活函数ReLU：
# - ReLU（Rectified Linear Unit）激活函数能够将负数值转化为零，增加模型的非线性，帮助捕捉复杂的特征
model.add(Activation('relu'))

# 第二个ConvLSTM2D层：
# - 这层与第一个ConvLSTM2D层类似，不过没有设置input_shape参数，因为它会自动推断
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', 
                     return_sequences=True))

# 输出层（Conv3D层）：
# - filters=3表示输出空间的维度为3，即每个时刻的预测类别数量（这里是3个类别）
# - kernel_size=(1, 1, 1)表示使用1x1x1的卷积核，实际上就是对每个时刻的预测进行处理（不改变空间维度）
# - activation='softmax'表示使用softmax激活函数，将输出转化为概率分布，适用于多分类问题
model.add(Conv3D(filters=3, kernel_size=(1, 1, 1), activation='softmax'))

# 编译模型：
# - optimizer='adam'表示使用Adam优化器，它是一种自适应学习率的优化算法，能有效地应对稀疏梯度问题
# - loss='categorical_crossentropy'表示使用多类别交叉熵作为损失函数，适用于多类分类问题
# - metrics=['accuracy']表示训练和测试时我们关注准确率这一评价指标
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 输出模型的摘要信息
model.summary()
