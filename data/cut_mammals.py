import csv

distinctList = []

dictf = open('../data/mammal_index_label.txt', 'r')
mammalDict = {}
mammalDictReverse = {}
while True:
    line = dictf.readline()
    if line:
        line = line.strip()
        line_split = line.split()
        if len(line_split) == 2:
            mammalDict[line_split[1]] = line_split[0]
            mammalDictReverse[line_split[0]] = line_split[1]
    else:
        break



tree = open('../data/tree2_mammal', 'r')
treeSet = set()
while True:
    branch = tree.readline()
    if branch:

        branch = branch.strip()
        line_split = branch.split()

        if len(line_split) == 2:

            childindex = line_split[1]
            parentindex = line_split[0]

            if parentindex == '1180':
                continue

            childname = mammalDictReverse[childindex]
            parentname = mammalDictReverse[parentindex]

            treeSet.add((childname,parentname))
    else:
        break

final = []

with open('../data/mammal_closure.csv', 'r') as csvf:
    reader = csv.reader(csvf)
    for row in reader:
        child = row[0]
        parent = row[1]
        weight = row[2]
        if (child,parent) in treeSet:
            final.append([child,parent,weight])

print(final)
with open('../data/mammal_closure_filtered.csv', 'w')as f:
    f_csv = csv.writer(f)
    f_csv.writerow(['id1','id2','weight'])
    f_csv.writerows(final)
