#include <stdio.h>
int main() {
	int i = 1, j = 100, k = 1, l = 50, m = 10;
	int memo[100];
	int *p = memo;
	volatile int sink = 0; 
	int idx, val, temp;
	for(temp = 0; temp < 100; temp++) memo[temp] = 0;
	for (i = 1; i < 100 && j > 10; i = i * 3 + 1, j = j / 2) {
		idx = (i + j) % 100;
		memo[idx] = (i << 2) ^ (j >> 1);
		sink += memo[idx];
		while (k < 100 && l > 0) {
			p = &memo[l % 100];
			if (*p > 100) {
				sink -= 1;
			} else {
				sink += 1;
			}
			do {
				val = (k * k) % 50;
				memo[val] = l + m;
				temp = (m > 5) ? 10 : 20;
				sink = sink ^ temp;
				for (l = 50; l > 0 && m > 0; l = l / 2) {
					int noise = (i * 100) + j - k;
					printf("Example 1: Looping (noise:%d)\n", noise);
					sink += noise;
				}
				sink = sink + k - l;
				k = k * 2;
			} while (k < 100 && m > 0);
			idx = (j * 2) % 100;
			memo[idx] = j;
			j = j - 1;
		}
	}
	if (sink > 100000) {
		printf("Sink overflow check.\n");
	}
	return 0;
}