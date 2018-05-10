def fib(n):
    a=[]
    a.append(0)
    a.append(1)
    for x in range(2,n+1):
        a.append(a[x-1]+a[x-2])
    return a

n=input("input")
print(fib(int(n)))
