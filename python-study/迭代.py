def findMinAndMax(L):
    if not L:  # 如果列表为空，返回 (None, None)
        return (None, None)
    
    min_value = L[0]  # 初始化最小值为第一个元素
    max_value = L[0]  # 初始化最大值为第一个元素

    for num in L:
        if num < min_value:
            min_value = num  # 更新最小值
        if num > max_value:
            max_value = num  # 更新最大值

    return (min_value, max_value)  # 返回元组

# 测试
if findMinAndMax([]) != (None, None):
    print('测试失败!')
elif findMinAndMax([7]) != (7, 7):
    print('测试失败!')
elif findMinAndMax([7, 1]) != (1, 7):
    print('测试失败!')
elif findMinAndMax([7, 1, 3, 9, 5]) != (1, 9):
    print('测试失败!')
else:
    print('测试成功!')
