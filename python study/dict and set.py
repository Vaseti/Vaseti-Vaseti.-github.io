#字典dict
d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
print(d['Michael'])

#set
#创建一个set，用{x,y,z,...}列出每个元素
s = {1, 2, 3}
#或者提供一个list作为输入集合
s = set([1, 2, 3])
#重复元素在set中自动被过滤
s = {1, 1, 2, 2, 3, 3}
#通过add(key)方法可以添加元素到set中，可以重复添加，但不会有效果
s.add(4)
#通过remove(key)方法可以删除元素
s.remove(4)
#set可以看成数学意义上的无序和无重复元素的集合，因此，两个set可以做数学意义上的交集、并集等操作
s1 = {1, 2, 3}
s2 = {2, 3, 4}
m=s1 & s2
print(m)
n=s1 | s2
print(n)