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
#include <emmintrin.h>
#include <omp.h>
#include <mpi.h>

int num_cpus;
int iters;
int width, height, whole_len;
int required_len;
double left, right, lower, upper;
double unit_x, unit_y;
int *fullImage;
int *tmp;
int *localImage;
omp_lock_t posLock;


bool get_position(long int & process_whole_len, long int &process_start, long int &process_end, long int &thread_start, long int &thread_end) {
    if (process_end >= process_whole_len) {
        return false;
    } else {
        process_start = process_end;
        process_end += required_len;
        if (process_end > process_whole_len) 
            process_end = process_whole_len;
        thread_start = process_start;
        thread_end = process_end;
        return true;
    }
}

void write_png(const char *filename, int iters, int width, int height, const int *buffer);


int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    num_cpus = CPU_COUNT(&cpu_set);

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
    unit_x = (right - left) / width;
    unit_y = (upper - lower) / height;
    required_len = 20;

    /* MPI */
    int rank, size;

    // MPI Init
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // since the data is 1D, we split the data by rank + size * i
    // mygodimatomato : did not handle the situation of size > whole_len, don't think it will happen
    // 13/4 = 3 -> every process will have 3 units of data
    // 13%4 = 1 
    // 0, 1, 2, 3
    // 0, 4, 8, 12 -> required_len = 4
    // 1, 5, 9
    // 2, 6, 10
    // 3, 7, 11
    long int process_whole_len = whole_len / size;
    if (whole_len % size > rank) {
        process_whole_len += 1;
    }
    
    // shared by all the processes, this is the cursor for run 
    long int process_start = 0;
    long int process_end = 0;
    
    
    // Image array initialization
    // rank 0 is responsible for the full image maintainence
    if (rank == 0) {
        fullImage = (int*)malloc(whole_len * sizeof(int));
        assert(fullImage);
    }

    localImage = (int*)malloc(process_whole_len * sizeof(int));
    tmp = (int*)malloc(process_whole_len * sizeof(int));

    /* mandelbrot set */
    // start pthread parallel
    #pragma omp parallel num_threads(num_cpus)
    {
        double x0, y0, x, y, length_squared;
        int repeats;
        double temp;
        long int thread_start, thread_end;
        while (true) {

            omp_set_lock(&posLock);
            if (get_position(process_whole_len, process_start, process_end, thread_start, thread_end) == false) {
                omp_unset_lock(&posLock);
                break;
            }
            omp_unset_lock(&posLock);
            // printf("thread %d, start from %d, end at %d\n", rank, thread_start, thread_end);
            for(int i = thread_start; i < thread_end; i++) {
            
                int real_pos = i * size + rank;
                x0 = (real_pos % width) * unit_x + left;
                y0 = (real_pos / width) * unit_y + lower;

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
                localImage[i] = rank;
            }
        }
    }


    int recvcounts[size];
    int displacements[size] = {0};
    for (int i = 0; i < size; i++){
        recvcounts[i] = whole_len/size;
        if (whole_len % size > i)
            recvcounts[i]++;
        if (i > 0){
            displacements[i] = displacements[i-1] + recvcounts[i];
        }
    }

    // if (rank == 0) {
    //     for (int i = 0; i < size; i++)
    //         printf("%d ", recvcounts[i]);
    //     printf("\n");
    //     for (int i = 0; i < size; i++)
    //         printf("%d ", displacements[i]);
    //     printf("\n");
    // }

    // printf("thread %d, len = %d\n", rank, process_whole_len);
    // send the data back to rank 0
    MPI_Gatherv(localImage, process_whole_len, MPI_INT, tmp, recvcounts, displacements, MPI_INT, 0, MPI_COMM_WORLD);
    // free(localImage);

    /* draw and cleanup */
    if (rank == 0) {
        long int now = whole_len / size;
        for (int i = 0; i < size; i++){
            for (int j = 0; j < whole_len/size; j++){
                fullImage[j]
            }
        }
        for(int i = 0; i < now; i++){
            fullImage[i*3] = tmp[i]
        }
        for (int i = 0; i < whole_len; i++) {
            printf("%d\n", fullImage[i]);
        }
        write_png(filename, iters, width, height, fullImage);
    }
    MPI_Finalize();
    free(fullImage);
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
