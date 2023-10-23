#include <assert.h>
#include <math.h>
#include <stdio.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	
  unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);

  unsigned long long x = 0;
  unsigned long long y = r;
  unsigned long long d = 1-r;
  unsigned long long count = y;

  while (x < y){
    x += 1;
    if (d < 0){
      d += 2 * x + 1;
    } else {
      y -= 1;
      d += 2 * (x - y) + 1;
    }
    count += y;
    count = count%k;
  }
  printf("%llu\n", (4 * count) % k);
	return 0;
}


