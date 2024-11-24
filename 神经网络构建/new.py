# 使用 Python 实现一个简单的神经网络
# 基于《Make Your Own Neural Network》by Tariq Rashid


import numpy
# 使用 scipy.special 提供的 sigmoid 函数 expit()
import scipy.special


# 定义神经网络类
class neuralNetwork:
    """
    该类实现了一个简单的前馈神经网络，支持单隐藏层的初始化、训练和查询操作。
    """

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        """
        初始化神经网络的结构和参数。

        :param inputnodes: 输入层节点数
        :param hiddennodes: 隐藏层节点数
        :param outputnodes: 输出层节点数
        :param learningrate: 学习率，用于控制权重更新幅度
        """
        # 设置各层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 初始化权重矩阵
        # wih 是输入层到隐藏层的权重，维度为 (hiddennodes, inputnodes)
        # who 是隐藏层到输出层的权重，维度为 (outputnodes, hiddennodes)
        # 权重使用正态分布初始化，均值为 0，标准差为 1/sqrt(输入节点数)
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # 设置学习率
        self.lr = learningrate

        # 激活函数：使用 sigmoid 函数 f(x) = 1 / (1 + e^(-x))
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        """
        使用目标值训练神经网络。

        :param inputs_list: 输入数据列表
        :param targets_list: 目标值数据列表
        """
        # 将输入数据和目标值转换为二维列向量
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 前向传播计算隐藏层信号
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # 前向传播计算输出层信号
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算输出误差：目标值 - 实际输出
        output_errors = targets - final_outputs
        # 计算隐藏层误差：由输出误差反传到隐藏层
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 更新隐藏层到输出层的权重
        self.who += self.lr * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs)
        )

        # 更新输入层到隐藏层的权重
        self.wih += self.lr * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs)
        )

    def query(self, inputs_list):
        """
        对于给定输入进行前向传播，计算输出。

        :param inputs_list: 输入数据列表
        :return: 输出层的激活值（结果）
        """
        # 将输入列表转换为二维列向量
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 计算隐藏层的输入和输出信号
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算输出层的输入和输出信号
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# 配置神经网络的结构
input_nodes = 3  # 输入层节点数
hidden_nodes = 3  # 隐藏层节点数
output_nodes = 3  # 输出层节点数

# 设置学习率
learning_rate = 0.3

# 创建神经网络实例
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 测试 query 方法，输入为 [1.0, 0.5, -1.5]
# 通过神经网络计算得到输出层的结果
result = n.query([1.0, 0.5, -1.5])
print("神经网络输出结果:", result)



