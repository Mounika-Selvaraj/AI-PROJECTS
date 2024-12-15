def add(a,b):
    return a+b
def sub(a,b):
    return a-b
def mul(a,b):
    return a*b
def div(a,b):
    if b==0:
        return "Error! Devision by Zero is Not Possible."
    else:
        return a/b
def calculator():
    print("Select the operation:")
    print("1. Add\n2. Subtraction\n3. Multiplication\n4. Division")
    
    while True:
        choice=input("Enter choices (1/2/3/4): ")
        if choice in ['1','2','3','4']:
            n1=float(input("Enter the First Number: "))
            n2=float(input("Enter the Second Number: "))
            
            if choice == '1':
                print(f"{n1}+{n2}={int(add(n1,n2))}") 
            if choice == '2':
                print(f"{n1}-{n2}={int(sub(n1,n2))}") 
            if choice == '3':
                print(f"{n1}+{n2}={int(mul(n1,n2))}") 
            if choice == '4':
                print(f"{n1}+{n2}={div(n1,n2)}") 
                
        next_calculation=input("Do you want to continue the next calculation? (Yes/No): ")
        if next_calculation.lower() == 'no':
            break
    print("Exiting the operation.......... Good Bye!!")

calculator()
        
    


