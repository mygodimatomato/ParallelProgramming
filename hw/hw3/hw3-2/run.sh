make && ./hw3-2 /tmp/dataset-nthu-pp23/pp23/share/hw3-2/cases/c04.1  ./out_1 > hw3_2_std_out.m
# g++ seq.cc -o seq && ./seq /tmp/dataset-nthu-pp23/pp23/share/hw3-2/cases/c03.1  ./out > seq_out.m
./modify_factor /tmp/dataset-nthu-pp23/pp23/share/hw3-2/cases/c04.1  ./out_2 > test_std_out.m
diff ./test_std_out.m ./hw3_2_std_out.m
diff ./out_1 ./out_2