# Common Compiler and Flags
CXX       := g++
NVCC      := nvcc
CXXFLAGS  := -std=c++11 -O3 -pthread -fopenmp
CFLAGS    := -O3 -lm -pthread -fopenmp
NVFLAGS   := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler="-fopenmp"
LDFLAGS   := -lm -lgomp

# Targets for Each Makefile
EXES1     := hw3-3
EXES2     := hw3-2
EXES3     := hw3-1

# All Targets
EXES      := $(EXES1) $(EXES2) $(EXES3)

.PHONY: all
all: $(EXES)

.PHONY: clean
clean:
	rm -f $(EXES)

# Rules for First Makefile
$(EXES1): hw3-3.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

# Rules for Second Makefile
$(EXES2): hw3-2.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

# Rules for Third Makefile
$(EXES3): hw3-1.cc
	$(CXX) $(CXXFLAGS) $(CFLAGS) -o $@ $?

# Additional rules or dependencies can be added here if needed
