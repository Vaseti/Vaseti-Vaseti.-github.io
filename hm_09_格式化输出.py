#定义字符串变量name，输出 我的名字叫 小明，请多多关照!  %s字符串
name="大小明"
print("我的名字叫%s，请多多关照!" %name)

#定义整数变量 student_no,输出 我的学号是000001  %06d,d整数，06是位数
student_no =1
print("我的学号是%06d" %student_no)

#定义小数price、weight、money,输出 苹果单价 9.00元/斤，购买了5.00斤,需要支付45.00元  %.2f f为浮点数，.2表示位数
price=8.5
weight=7.5
money=price*weight
aaa ="苹果单价 %.2f 元/斤，购买了 %.2f 斤,需要支付 %.2f 元" % (price, weight, money)
print(aaa)
#定义一个小数scale，输出 数据比例为10.00%
scale=0.25
bbb="数据比例为%.2f%%" %(scale*100)
print(bbb)