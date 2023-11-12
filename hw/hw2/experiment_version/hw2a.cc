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
#include <emmintrin.h>

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
    double tmp[2];
    long int my_start, my_end, now_0, now_1; 
    int i;
    __m128d vec_x0, vec_y0, vec_x, vec_y, vec_length_squared, vec_temp, vec_repeats, vec_x_squared, vec_y_squared, vec_xy, mask;
    __m128d vec_ones = _mm_set1_pd(1);
    __m128d vec_four = _mm_set1_pd(4);
    __m128d vec_iters = _mm_set1_pd(iters);
    int maskResult = 0;
    // run a loop to get next position and calculate it's mandelbrot
    while (true) {
        pthread_mutex_lock(&mutex);
        if (get_position(my_start, my_end) == false) {
            pthread_mutex_unlock(&mutex);
            break;
        }
        pthread_mutex_unlock(&mutex);
        // initialize flag, 0 means the vector is empty
        i = my_start;
        // calculate the mandelbrot
        // for (int i = my_start; i < my_end; i++)
        // printf("%d \n", i);
        while (true) {
            if (i >= my_end) break;
            if (maskResult == 0) {
                now_0 = i;
                i++;
                now_1 = i;
                i++;
                double x_0 = (now_0 % width) * unit_x + left;
                double x_1 = (now_1 % width) * unit_x + left;
                double xx[2] = {x_0, x_1};
                vec_x0 = _mm_load_pd(&xx[0]); // I want the vector arrange as [now_0, now_1]
                double y_0 = (now_0 / width) * unit_y + lower;
                double y_1 = (now_1 / width) * unit_y + lower;
                double yy[2] = {y_0, y_1};
                vec_y0 = _mm_load_pd(&yy[0]);
                vec_repeats = _mm_set1_pd(0);
                vec_x = _mm_set1_pd(0);
                vec_y = _mm_set1_pd(0);
                vec_length_squared = _mm_set1_pd(0);
                // double tmp1[2];
                // double tmp2[2];
                // _mm_storeu_pd((double*)&tmp1, vec_x0);
                // _mm_storeu_pd((double*)&tmp2, vec_y0);
                // printf("condition 1, i = %d, x, y= (%f, %f, %f, %f), origin is (%f, %f, %f, %f)\n", i, tmp1[0], tmp1[1], tmp2[0], tmp2[1], xx[0], xx[1], yy[0], yy[1]);
            } else if (maskResult == 1) { // high element need to change
                now_1 = i++;
                double xx[1] = {(now_1 % width) * unit_x + left};
                vec_x0 = _mm_loadh_pd(vec_x0, &xx[0]);
                double yy[1] = {(now_1 / width) * unit_y + lower};
                vec_y0 = _mm_loadh_pd(vec_y0, &yy[0]);
                double zero[1]= {0};
                vec_repeats = _mm_loadh_pd(vec_repeats, &zero[0]);
                vec_x = _mm_loadh_pd(vec_x, &zero[0]);
                vec_y = _mm_loadh_pd(vec_y, &zero[0]);
                vec_length_squared = _mm_loadh_pd(vec_length_squared, &zero[0]);
                // double tmp1[2];
                // double tmp2[2];
                // _mm_storeu_pd((double*)&tmp1, vec_x0);
                // _mm_storeu_pd((double*)&tmp2, vec_y0);
                // printf("condition 2, i = %d, x, y= (%f, %f, %f, %f), origin is (%f, %f)\n", i, tmp1[0], tmp1[1], tmp2[0], tmp2[1], xx[0], yy[0]);
                // _mm_storeu_pd((double*)&tmp1, vec_x);
                // _mm_storeu_pd((double*)&tmp2, vec_y);
                // printf("condition 2, i = %d, x, y= (%f, %f, %f, %f)\n", i, tmp1[0], tmp1[1], tmp2[0], tmp2[1]);
            } else if (maskResult == 2) { // low element need to change
                // printf("condition 3, i = %d\n", i);
                now_0 = i++;
                double xx[1] = {(now_0 % width) * unit_x + left};
                vec_x0 = _mm_loadl_pd(vec_x0, &xx[0]);
                double yy[1] = {(now_0 / width) * unit_y + lower};
                vec_y0 = _mm_loadl_pd(vec_y0, &yy[0]);
                double zero[1] = {0};
                vec_repeats = _mm_loadl_pd(vec_repeats, &zero[0]);
                vec_x = _mm_loadl_pd(vec_x, &zero[0]);
                vec_y = _mm_loadl_pd(vec_y, &zero[0]);
                vec_length_squared = _mm_loadl_pd(vec_length_squared, &zero[0]);
                // double tmp1[2];
                // double tmp2[2];
                // _mm_storeu_pd((double*)&tmp1, vec_x);
                // _mm_storeu_pd((double*)&tmp2, vec_y);
                // printf("condition 3, i = %d, x, y= (%f, %f, %f, %f)\n", i, tmp1[0], tmp1[1], tmp2[0], tmp2[1]);
                // int tmp3[2];
                // _mm_storeu_pd((double*)&tmp3, vec_repeats);
                // printf("condition 3, i = %d, %d\n", i, tmp3[0]);
            }
            double aaa[2];
            _mm_storeu_pd((double*)&aaa, vec_ones);
            // printf("-----------------------------------------\n");
            // printf("checking ones %d, %f \n", aaa[0], aaa[1]);
            while (true) {
                // while (repeats < iters && length_squared < 4)

                // temp = (x*x) - (y*y) + x0;
                vec_x_squared = _mm_mul_pd(vec_x, vec_x);
                vec_y_squared = _mm_mul_pd(vec_y, vec_y);
                vec_temp = _mm_add_pd(_mm_sub_pd(vec_x_squared, vec_y_squared), vec_x0);

                // y = 2 * x * y + y0; 
                vec_xy = _mm_mul_pd(vec_x, vec_y);
                vec_y = _mm_add_pd(_mm_mul_pd(vec_xy, _mm_set1_pd(2)), vec_y0);

                // x = temp;
                vec_x = vec_temp;

                // length_squared = x * x + y * y;
                vec_x_squared = _mm_mul_pd(vec_x, vec_x);
                vec_y_squared = _mm_mul_pd(vec_y, vec_y);
                vec_length_squared = _mm_add_pd(vec_x_squared, vec_y_squared);

                // ++repeats;
                vec_repeats = _mm_add_pd(vec_repeats, vec_ones);
                __m128d vec_tmp = vec_repeats;

                // while (repeats < iters && length_squared < 4)
                mask = _mm_and_pd(_mm_cmplt_pd(vec_length_squared, vec_four), _mm_cmplt_pd(vec_tmp, _mm_set1_pd(iters)));
                maskResult = _mm_movemask_pd(mask);
                if (maskResult <= 2) break; // If any one value in mask is 0 -> break
            }
            
            // image[i] = repeats;
            double res[2];
            _mm_storeu_pd((double*)&res, vec_repeats);
            if (maskResult == 0) { // both vec elem cause break
                image[now_0] = (int)res[0];
                image[now_1] = (int)res[1];
                // printf("condition1\n");
                // printf("i = %d, %d\n", now_0, (int)res[0]);
                // printf("i = %d, %d\n", now_1, (int)res[1]);
            } else if (maskResult == 1) { // Lower element is non-zero, higher element is zero
                // printf("condition2\n");
                image[now_1] = (int)res[1];
                // printf("i = %d, %d\n", now_1, (int)res[1]);
            } else if (maskResult == 2) {
                image[now_0] = (int)res[0];
                // printf("condition3\n");
                // printf("i = %d, %d\n", now_0, (int)res[0]);
            }
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


    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    #pragma GCC ivdep
    for (int i = 0; i < num_cpus; i++) {
        pthread_create(&threads[i], NULL, cal_mandelbrot, NULL);
    }

    // stop the threads
    #pragma GCC ivdep
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

