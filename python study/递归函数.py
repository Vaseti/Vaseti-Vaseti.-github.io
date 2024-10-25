def move(n, a, b, c):
    if n == 1:
        print(a, '-->', c)  # 移动一个盘子
    else:
        move(n - 1, a, c, b)  # Step 1: A -> B
        print(a, '-->', c)     # Step 2: A -> C
        move(n - 1, b, a, c)  # Step 3: B -> C

# 调用函数，打印从A移动到C的步骤
move(3, 'A', 'B', 'C')
