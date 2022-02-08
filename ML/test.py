class bank:
 #   acc = 9090
    def __init__(self, x, y, z):
        self.name = x
        self.acc = y
        self.bName = z
    def printdetails(self):
        print (f"Name is {self.name} Account number is {self.acc} Bank name is {self.bName}")

# b = bank()
# print(b.acc)  

b = bank("Nithesh", 9090, "SBI")
c = bank("Rohan", 8888, "HDFC")

b.printdetails()
c.printdetails()