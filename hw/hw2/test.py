whole_len = 13
size = 4
recvcounts = [4, 3, 3, 3]
displacements = [0, 4, 7, 10]
test = [0, 4, 8, 12,1, 5, 9,2,6, 10, 3, 7, 11]
test_2 = []

# use size, recvcounts, displacement to get the sorted test_2, which should be [0,1,2,3,4,5,6,7,8,9,10,11,12]
# append test_2 with the sorted numbers
for j in range(3):
  for i in range(size):
    test_2.append(test[displacements[i] + j])
for i in range(whole_len%size):
  test_2.append(test[displacements[i] + 3] )

for i in range(whole_len):
  print(test_2[i])