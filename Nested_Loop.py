x =int(input("masukkan Jumlah baris:"))
for i in range(x):
    for j in range (i+1):
        print("*",end="")
    print()


    
for i in reversed(range(x-1)):
    for j in range (i+1):
        print("*",end="")
    print() 
for i in range(x):
    print(("*" *(1+2*i)).center(1+2*x))
for i in reversed(range(x)):
    print(("*" *(1+2*i)).center(1+2*x))
