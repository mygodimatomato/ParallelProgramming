#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

int iters;
double left, right;
double lower, upper;
int width, height;
int num_cpus;
double unit_x, unit_y;
long int whole_len, start, end;
int required_len;
int* image;


pthread_mutex_t mutex;

bool get_position(long int &my_start, long int &my_end) {
    if (end >= whole_len) {
        return false;
    } else {
        start = end;
        end += required_len;
        if (end > whole_len) 
            end = whole_len;
        my_start = start;
        my_end = end;
        return true;
    }
}


void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void* cal_mandelbrot(void* arg) {
    long int my_start, my_end; 
    double x0, y0;
    double x, y, length_squared;
    int repeats;
    double temp;
    
    // run a loop to get next position and calculate it's mandelbrot
    while (true) {
        pthread_mutex_lock(&mutex);
        if (get_position(my_start, my_end) == false) {
            pthread_mutex_unlock(&mutex);
            break;
        }
        // printf("%d\n", my_end);
        pthread_mutex_unlock(&mutex);

        // calculate the mandelbrot
        
        for (int i = my_start; i < my_end; i++) {

            x0 = (i % width) * unit_x + left; // not sure, need check
            y0 = (i / width) * unit_y + lower; // not sure, need check

            repeats = 0;
            x = 0;
            y = 0;
            length_squared = 0;
            while (repeats < iters && length_squared < 4) {
                temp = (x*x) - (y*y) + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;

            }
            printf("i = %d, %d\n", i, repeats);
            image[i] = repeats;
        }
    }
    return NULL;
}



int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    num_cpus = CPU_COUNT(&cpu_set);
    // setting up the pthread pool
    pthread_t threads[num_cpus]; // this is for the thread pool
    unsigned long long ID[num_cpus]; // this is for the data that pass into the thread

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    whole_len = width * height;
    required_len = 20;
    start = 0;
    end = 0;
    unit_x = (right - left) / width;
    unit_y = (upper - lower) / height;
    // printf("%d \n", whole_len);

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    for (int i = 0; i < num_cpus; i++) {
        pthread_create(&threads[i], NULL, cal_mandelbrot, NULL);
    }

    // stop the threads
    for (int i = 0; i < num_cpus; i++) {
        pthread_join(threads[i], NULL);
    }

    // free the mutex
    pthread_mutex_destroy(&mutex);

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);

    return 0;
}

