classmates=['Mike','Xiaoming','Xiaohong','Jock','Jonh']
#正序输出
n=0
while n<len(classmates):
    print('Hello,'+classmates[n])
    n+=1
#倒序输出
m=len(classmates)
while m>0:
    m-=1
    print('This is a good day,'+classmates[m])
#对列表进行修改
classmates.append('Whitch')
classmates.insert(0,'Pike')
classmates.pop(2)
classmates[3]='Black'
print(classmates)