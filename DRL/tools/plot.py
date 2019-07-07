import matplotlib.pyplot as plt
import sys

# with open("plot.txt") as f:
#     cont = f.readlines()
#     num = []
#     for i in cont:
#         num.append(float(i))
#     print(num)
#     plt.plot(num)
#     plt.show()

def read_log_file(filename):
    try:
        with open(filename) as f:
            cont = f.readlines()
            if len(cont) == 1:
                cont = cont[0].split()
                # print(cont)
            
            num_list = [float(i) for i in cont]
            
            return num_list

    except FileExistsError:
        print("file %s is not exist" % filename)
        return None

if __name__ =="__main__":
    # print("succ")
    args = sys.argv[1:] # file list
    
    # check file valid
    for filename in args:
        cont = read_log_file(filename)
        plt.plot(cont)
    plt.legend(args)
    plt.show()
