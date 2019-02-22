import numpy as np
from scipy.spatial import distance as dt

self_pt = []  # Self points all have radius R
detector = []  # Detector Point Co-ordinates
detector_rad = []  # Detector Radius
test_pt = []  # Test Point Co-ordinates
test_label = []  # Test Point labels True for 0 and False for 1 and 2
all_pt = []  # All the data points

f_ins = open("NSA_instructions.txt", "r")
lns_ins = f_ins.readlines()
R, M, no_of_samples = [int(x) for x in lns_ins[0].strip(" \n").split(" ")]
no_of_epoch = int(lns_ins[1].strip(" \n").strip(" "))

print("Self Radius: " + str(R) + " Detector No: " + str(M))

f_in = open("data_input_train.txt", "r")
f_out = open("data_output_train.txt", "r")
lns_in = f_in.readlines()
c = p = 0
for lns in f_out:
    intgs = [int(x) for x in lns.strip(' \n').split(' ')]
    all_pt.append(np.array([int(x) for x in lns_in[c].strip(' \n').split(' ')]))
    if (intgs[0] == 1):
        arr = [int(x) for x in lns_in[c].strip(' \n').split(' ')]
        self_pt.append(np.array(arr))
        p += 1  # number of zeros
    c += 1  # no of points
print("Total no of zeros " + str(p))
f_in.close()
f_out.close()

f_in = open("data_input_test.txt", "r")
f_out = open("data_output_test.txt", "r")
lns_in = f_in.readlines()
c = 0
for lns in f_out:
    intgs_in = [int(x) for x in lns_in[c].strip(' \n').split(' ')]
    intgs_out = [int(x) for x in lns.strip(' \n').split(' ')]
    c += 1
    test_pt.append(intgs_in)
    if (intgs_out[0] == 1):
        test_label.append(True)
    else:
        test_label.append(False)
f_in.close()
f_out.close()

dim = len(self_pt[0])  # Dimension of the Self Points


def distance(x1, x2):  # Minkowski 1 distance between two points
    return dt.minkowski(x1, x2, 1) / len(x1)


def calcR(x):
    # Finding the distance of sample point nearest to the detector point
    min_dist = float("inf")
    for pt in self_pt:
        dist = distance(x, pt)
        if (dist < min_dist):
            min_dist = dist

    # Checking if the point lies within the range of any detector point
    pos = None  # mark the detector which is near
    for i in range(len(detector)):
        dist = distance(x, detector[i])
        if (dist < detector_rad[i]):
            return -1
        if (dist < min_dist):
            min_dist = dist
            pos = i

    # Calculating the radius. If it is a valid detector then r > 0
    if pos is not None:
        r = min_dist - detector_rad[pos]  # calculating distance when detector point is near
    else:
        r = min_dist - R  # calculating distance when source point is near

    return r


max_values = np.amax(np.array(all_pt), axis=0)  # finding the maximum value of each field


class GA:
    def __init__(self):
        gene_T = []
        for i in range(dim):
            gene_T.append(np.random.random_integers(1, max_values[i], size=no_of_samples))
        self.population = np.transpose(gene_T)
        # Rectifying the generated population
        for p in range(no_of_samples):
            r = calcR(self.population[p])  # Radius of the possible detector point
            while (r <= 0):  # Rectifying until valid r>0 is found
                i = 0
                while (i < dim):
                    self.population[p][i] = np.random.randint(1, max_values[i])
                    i += 1
                r = calcR(self.population[p])

    def mutate(self, population):  # performing the mutation of the newly created population
        total_no = dim * int(no_of_samples / 2)
        no_of_mutation = int(.4 * total_no)  # 30% mutation is done
        arr = np.array([1] * no_of_mutation + [0] * (total_no - no_of_mutation))
        np.random.shuffle(arr)
        arr = np.reshape(arr, (int(no_of_samples / 2), dim))
        indices = np.argwhere(arr == 1)  # finding the places for mutation randomly
        for i in indices:
            population[i[0]][i[1]] = np.random.randint(1, max_values[i[1]])  # providing the new values
        return population

    def crossover(self, cost_lst):
        median = np.median(cost_lst)  # used to find the samples whose
        new_population = np.zeros((int(no_of_samples / 2), dim), dtype=int)
        for i in range(0, int(no_of_samples / 2)):
            parent1 = self.population[i * 2]
            parent2 = self.population[i * 2 + 1]
            # performing multi-point crossover odd places from 2nd parent and others from first
            for j in range(dim):
                if (j % 2 == 0):
                    new_population[i][j] = parent1[j]
                else:
                    new_population[i][j] = parent2[j]
        new_population = self.mutate(new_population)
        # Rectifying the generated population
        for p in range(int(no_of_samples / 2)):
            r = calcR(new_population[p])  # Radius of the possible detector point
            while (r <= 0):  # Rectifying until valid r>0 is found
                i = 0
                while (i < dim):
                    new_population[p][i] = np.random.randint(1, max_values[i])
                    i += 1
                r = calcR(new_population[p])
        # Replacing the bad populations
        c = int(no_of_samples / 2) - 1
        for i in range(len(self.population)):
            if (cost_lst[i] < median and c >= 0):
                self.population[i] = new_population[c]
                c -= 1
            if (c < 0):
                break


def create_negative_detectors():
    obj = GA()
    cost_lst = []
    for i in range(no_of_epoch):
        cost_lst = []
        for j in range(no_of_samples):
            cost_lst.append(calcR(obj.population[j]))
        obj.crossover(cost_lst)
        # print(str(max(cost_lst))+" ",end="")
    pos = np.arange(no_of_samples)
    pos = [x for _, x in sorted(zip(cost_lst, pos), reverse=True)]
    pos = pos[:2]
    # print(obj.population)
    return (obj.population[pos[0]], obj.population[pos[1]]), (cost_lst[pos[0]], cost_lst[pos[1]])


# print(create_negative_detectors())

def populate_detectors():
    i = 0
    while (i < M):
        population, pop_rad = create_negative_detectors()
        # Appending the best detector
        detector.append(population[0])
        detector_rad.append(pop_rad[0])
        i += 1
        print("INFO: Appending Detector " + str(i) + " " + " ".join(str(x) for x in population[0]) + " Radius: " + str(
            pop_rad[0]))

        if (i == M):
            break

        r = calcR(population[1])  # calculating the new radius for the 2nd Best detector
        if (r > 0):  # Appending if valid
            detector.append(population[1])
            detector_rad.append(r)
            i += 1
            print("INFO: Appending Detector " + str(i) + " " + " ".join(
                str(x) for x in population[1]) + " Radius: " + str(r))


def calcDecR(x):
    for i in range(M):
        dist = distance(detector[i], x)
        if (dist < detector_rad[i]):
            return False
    return True


def test():
    populate_detectors()
    acc = 0
    for i in range(len(test_pt)):
        pred = calcDecR(test_pt[i])
        if (pred == test_label[i]):
            acc += 1
        else:
            print("Wrong Prediction " + " ".join(str(x) for x in test_pt[i]) + " Predct: " + str(
                pred) + " Actual: " + str(test_label[i]))
    print("Accuracy " + str(acc/len(test_pt)))


test()
