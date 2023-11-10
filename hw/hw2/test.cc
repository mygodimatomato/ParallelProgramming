        for (i = my_start; i < my_end; i++) {
            if (flag[0] == 0 ) {
                vec_0_i = 
                tmp[0] = (i % width) * unit_x + left;
                vec_x0 = _mm_load_pd(tmp[0], tmp[1]);
            } 
            if (flag[1] == 0) {
                tmp[1] = ((i+1) % width) * unit_x + left;

            }
            tmp[0] = (i / width) * unit_y + lower;
            tmp[1] = ((i+1) / width) * unit_y + lower;
            vec_y0 = _mm_load_pd(tmp[0], tmp[1]);
            vec_repeats = _mm_set1_pd(0);
            vec_x = _mm_set1_pd(0);
            vec_y = _mm_set1_pd(0);
            vec_length_squared = _mm_set1_pd(0);
            vec_four = _mm_set1_pd(4);
            xvec_iters = _mm_set1_pd(iters);
            while (true) {
                if (){
                    break;
                } else {
                    vec_x_squared = _mm_mul_pd(vec_x, vec_x);
                    vec_y_squared = _mm_mul_pd(vec_y, vec_y);
                    vec_xy = _mm_mul_pd(vec_x, vec_y);
                    vec_temp = _mm_add_pd(_mm_sub_pd(vec_x_squared, vec_y_squared), vec_x0);
                    vec_y = _mm_add_pd(_mm_mul_pd(vec_xy, _mm_set1_pd(2)), vec_y0);
                    vec_x = vec_temp;
                }
            }
        }