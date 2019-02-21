

f_dat = open("Fisher.txt","r")
lns_dat = f_dat.readlines()

max_label = 3
no_of_train = 100
no_of_test = 50

f_in = open("data_input_train.txt","w")
f_out = open("data_output_train.txt","w")

for lns in lns_dat[:no_of_train]:
    intgs = [int(x) for x in lns.strip(' \n').split('\t')]
    for j in range(max_label):
        if(intgs[0] == j):
            f_out.write("1 ")
        else:
            f_out.write("0 ")
    f_out.write("\n")

    intgs = intgs[1:]
    f_in.write(" ".join(str(x) for x in intgs))
    f_in.write("\n")

f_in.close()
f_out.close()

f_in = open("data_input_test.txt","w")
f_out = open("data_output_test.txt","w")

for lns in lns_dat[no_of_train:]:
    intgs = [int(x) for x in lns.strip(' \n').split('\t')]
    for j in range(max_label):
        if(intgs[0] == j):
            f_out.write("1 ")
        else:
            f_out.write("0 ")
    f_out.write("\n")

    intgs = intgs[1:]
    f_in.write(" ".join(str(x) for x in intgs))
    f_in.write("\n")

f_in.close()
f_out.close()

