def trim(s):
    start = 0
    end = len(s)

    # 找到第一个非空字符的索引
    while start < end and s[start] == ' ':
        start += 1

    # 找到最后一个非空字符的索引
    while end > start and s[end - 1] == ' ':
        end -= 1

    # 使用切片提取去除空格后的字符串
    return s[start:end]

# 测试:
if trim('hello  ') != 'hello':
    print('测试失败!')
elif trim('  hello') != 'hello':
    print('测试失败!')
elif trim('  hello  ') != 'hello':
    print('测试失败!')
elif trim('  hello  world  ') != 'hello  world':
    print('测试失败!')
elif trim('') != '':
    print('测试失败!')
elif trim('    ') != '':
    print('测试失败!')
else:
    print('测试成功!')
