# Python代码：实现三层神经网络并对MNIST数据集进行学习和测试
# (c) Tariq Rashid, 2016
# 许可：GPLv2

import numpy  # 导入NumPy库，用于处理数组和矩阵运算
import scipy.special  # 导入SciPy的特殊函数模块，用于激活函数Sigmoid
import matplotlib.pyplot  # 导入Matplotlib库（尽管此处未使用，可用于可视化）

# 定义神经网络类
class neuralNetwork:
    
    # 初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        """
        :param inputnodes: 输入层的节点数
        :param hiddennodes: 隐藏层的节点数
        :param outputnodes: 输出层的节点数
        :param learningrate: 学习率，用于控制权重更新的步幅
        """
        # 设置输入、隐藏和输出层的节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # 初始化输入到隐藏层的权重矩阵 wih 和隐藏到输出层的权重矩阵 who
        # 权重遵循正态分布，均值为 0，标准差与节点数的倒数平方根相关
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # 学习率
        self.lr = learningrate
        
        # 激活函数为Sigmoid，定义为lambda匿名函数
        self.activation_function = lambda x: scipy.special.expit(x)  # Sigmoid函数公式为 1 / (1 + e^(-x))
    
    # 训练神经网络
    def train(self, inputs_list, targets_list):
        """
        训练神经网络：
        :param inputs_list: 输入值列表，代表样本特征
        :param targets_list: 目标值列表，代表对应的标签
        """
        # 将输入列表和目标值列表转换为2D数组，并转置为列向量
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # 前向传播：计算隐藏层的输入信号和输出信号
        hidden_inputs = numpy.dot(self.wih, inputs)  # 隐藏层接收到的信号
        hidden_outputs = self.activation_function(hidden_inputs)  # 隐藏层的激活输出
        
        # 前向传播：计算输出层的输入信号和输出信号
        final_inputs = numpy.dot(self.who, hidden_outputs)  # 输出层接收到的信号
        final_outputs = self.activation_function(final_inputs)  # 输出层的激活输出
        
        # 计算误差：
        output_errors = targets - final_outputs  # 输出误差=目标值-实际输出
        hidden_errors = numpy.dot(self.who.T, output_errors)  # 隐藏层误差从输出误差反向传播得到
        
        # 权重更新公式（基于梯度下降法）：
        # 更新隐藏层到输出层的权重
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        # 更新输入层到隐藏层的权重
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))
    
    # 查询神经网络
    def query(self, inputs_list):
        """
        前向传播过程：根据输入值查询神经网络输出
        :param inputs_list: 输入值列表，代表测试样本特征
        :return: 输出值，代表网络对输入的预测结果
        """
        # 将输入列表转换为2D数组，并转置为列向量
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # 计算隐藏层的输入和输出信号
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # 计算输出层的输入和输出信号
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

# 设置输入、隐藏和输出层的节点数
input_nodes = 784  # 输入层有784个节点（MNIST图像28x28像素）
hidden_nodes = 200  # 隐藏层设置200个节点
output_nodes = 10  # 输出层有10个节点（代表数字0到9的分类）

# 设置学习率
learning_rate = 0.1

# 创建神经网络实例
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 读取MNIST训练数据
training_data_file = open(r"D:\git warehouse\Vaseti-Vaseti.-github.io\神经网络构建\mnist_dataset\mnist_train_100.csv", 'r')  # 打开训练数据文件
training_data_list = training_data_file.readlines()  # 按行读取数据
training_data_file.close()

# 设置训练轮数
epochs = 5  # 训练数据重复使用5次
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')  # 分割每行数据
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # 归一化像素值到0.01-1.00范围
        targets = numpy.zeros(output_nodes) + 0.01  # 初始化目标值为0.01
        targets[int(all_values[0])] = 0.99  # 目标值对应标签的值设为0.99
        n.train(inputs, targets)  # 调用训练函数

# 读取MNIST测试数据
test_data_file = open(r"D:\git warehouse\Vaseti-Vaseti.-github.io\神经网络构建\mnist_dataset\mnist_train_100.csv", 'r')  # 打开测试数据文件
test_data_list = test_data_file.readlines()  # 按行读取数据
test_data_file.close()

# 测试神经网络
scorecard = []  # 初始化成绩单，用于记录每次测试结果
for record in test_data_list:
    all_values = record.split(',')  # 分割每行数据
    correct_label = int(all_values[0])  # 正确的标签为行首值
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # 归一化像素值
    outputs = n.query(inputs)  # 调用查询函数
    label = numpy.argmax(outputs)  # 获取网络预测的标签
    scorecard.append(1 if label == correct_label else 0)  # 正确预测记为1，错误预测记为0

# 计算性能评分
scorecard_array = numpy.asarray(scorecard)  # 转换为NumPy数组
print("performance = ", scorecard_array.sum() / scorecard_array.size)  # 输出正确率
