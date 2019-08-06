class test:
    def __init__(self):
        self.a = 1
    def create(self):
        para_name = {"dog":3, "my_dog":2, "your_dog":1}
        # locals()["dog"] = 1
        # print(self.dog)
        names = self.__dict__
        for i in para_name:
            names[i] = para_name[i]
        print(self.__dict__["dog"])
        

if __name__ == "__main__":
    agent = test()
    agent.create()
# locals()["dog"] = 1
# print(dog)