#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// usage: jtime m n program -parameters, will call program n times, then repeat this m times
int main(int argc, char *argv[]) {
    struct timespec start, end;
	char* endptr;
	errno = 0;
	
	// Code to be timed
	if (argc > 3) {
		intmax_t repeats = strtoimax(argv[1],&endptr,10);
		intmax_t loops = strtoimax(argv[2],&endptr,10);
		char buf[1024];
		buf[0] = '\0';
		for (int i = 3; i < argc && i < 20; i++) {
			strcat(buf, argv[i]);
			strcat(buf, " ");
		}
		for (int i = 0; i < repeats; i++) {
			// Start the timer
			clock_gettime(CLOCK_MONOTONIC, &start);
			for (int j = 0; j < loops; j++) {
				//printf("%s", buf);
				system(buf);
			}
			// Stop the timer
			clock_gettime(CLOCK_MONOTONIC, &end);

			// Calculate elapsed time in nanoseconds
			long long elapsed_ns = (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
			int minutes = (end.tv_sec - start.tv_sec) / 60;
			double seconds = (elapsed_ns / 1000000000.0) - minutes * 60;
			printf("\nreal    %dm%2.9fs\n", minutes, seconds);
		}
	}

    return 0;
}