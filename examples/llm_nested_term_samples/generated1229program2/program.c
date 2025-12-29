#include <stdio.h>
int main() {
	int a = 1, b = 100, c = 1, d = 10, e = 100, f = 50;
	int data[256];
	int *ptr = data;
	volatile int sink = 0;
	int i, temp, idx, val;
	for (i = 0; i < 256; i++) {
		data[i] = 0;
	}
	while (a < 50 && b > 5) {
		idx = (a * 13 + b * 7) % 256;
		data[idx] = (a << 2) ^ (b >> 1);
		sink += data[idx];
		a = a * 2;
		b = b / 2;
		val = (a + b) % 100;
		temp = (val > 50) ? 1 : 0;
		sink = sink + temp;
		for (c = 1; c < 100 && d > 0; c = c * 2, d = d / 2) {
			ptr = &data[(c + d) % 256];
			*ptr = *ptr ^ c;
			sink = sink ^ d;
			do {
				temp = (e > d) ? e : d;
				data[temp % 256] = data[temp % 256] + 1;
				sink = sink | temp;
				while (e > 0 && c < 50) {
					val = (e * 2) - c;
					if (val > 0) {
						sink = sink + 1;
					} else {
						sink = sink - 1;
					}
					for (f = 50; f > 0 && e > 0; f = f / 2) {
						idx = f % 256;
						data[idx] = f * f + sink;
						temp = data[idx] & 0xFF;
						sink = sink + temp;
						e = e / 2;
					}
				}
			} while (d > 0 && e < 100);
		}
	}
	if (sink != 0) {
		printf("End of program. Sink value: %d\n", sink);
	}
	return 0;
}