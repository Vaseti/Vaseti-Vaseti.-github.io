name=['aaa','bbb','ccc','ddd']
print(name)
name[1]='eee'
print(name)
name.append('fff')
print(name)
name.insert(0,'ggg')
print(name)
name.insert(2,'hhh')
print(name)
message='Only can invite two person,sorry.'
print(message)
while len(name)>2:
    new_name = name.pop()
    print(f"I am sorry that I can't invite you, {new_name}.")

print("Still invited:")
for person in name:
    print(person)

while len(name)>0:
    del name[0]
    print(name)




