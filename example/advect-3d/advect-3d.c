/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* advect-3d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "advect-3d.h"
/* Array initialization. */
// static
// void init_array (int n,
//       DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n),
//       DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
// {
//   int i, j, k;

//   for (i = 0; i < n; i++)
//     for (j = 0; j < n; j++)
//       for (k = 0; k < n; k++)
//         A[i][j][k] = B[i][j][k] = (DATA_TYPE) (i + j + (n-k))* 10 / (n);
// }

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
// static
// void print_array(int n,
//       DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n))

// {
//   int i, j, k;

//   POLYBENCH_DUMP_START;
//   POLYBENCH_DUMP_BEGIN("A");
//   for (i = 0; i < n; i++)
//     for (j = 0; j < n; j++)
//       for (k = 0; k < n; k++) {
//          if ((i * n * n + j * n + k) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
//          fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
//       }
//   POLYBENCH_DUMP_END("A");
//   POLYBENCH_DUMP_FINISH;
// }

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_advect_3d(int n,
                             DATA_TYPE POLYBENCH_3D(ab, N, N, N, n, n, n),
                             DATA_TYPE POLYBENCH_3D(al, N, N, N, n, n, n),
                             DATA_TYPE POLYBENCH_3D(af, N, N, N, n, n, n),
                             DATA_TYPE POLYBENCH_3D(a, N, N, N, n, n, n),
                             DATA_TYPE POLYBENCH_3D(athird, N, N, N, n, n, n),
                             DATA_TYPE POLYBENCH_3D(uyb, N, N, N, n, n, n),
                             DATA_TYPE POLYBENCH_3D(uxl, N, N, N, n, n, n),
                             DATA_TYPE POLYBENCH_3D(uzf, N, N, N, n, n, n))
{
    int i, j, k, m;
    DATA_TYPE temp;
#pragma scop


    for (j = 4; j <= _PB_N-1; j++)
        for (i = 4; i <= _PB_N-2; i++)
            for (k = 4; k <= _PB_N-2; k++)
                ab[j][i][k] = (0.2 * (a[j - 1][i][k] + a[j][i][k]) + 0.5 * (a[j - 2][i][k] + a[j + 1][i][k]) + 0.3 * (a[j - 3][i][k] + a[j + 2][i][k])) * 0.3 * uyb[j][i][k];

    for (j = 4; j <= _PB_N - 2; j++)
        for (i = 4; i <= _PB_N - 1; i++)
            for (k = 4; k <= _PB_N - 2; k++)
                al[j][i][k] = (0.2 * (a[j][i - 1][k] + a[j][i][k]) + 0.5 * (a[j][i - 2][k] + a[j][i + 1][k]) + 0.3 * (a[j][i - 3][k] + a[j][i + 2][k])) * 0.3 * uxl[j][i][k];

    for (j = 4; j <= _PB_N - 2; j++)
        for (i = 4; i <= _PB_N - 2; i++)
            for (k = 4; k <= _PB_N - 1; k++)
                af[j][i][k] = (0.2 * (a[j][i][k - 1] + a[j][i][k]) + 0.5 * (a[j][i][k - 2] + a[j][i][k + 1]) + 0.3 * (a[j][i][k - 3] + a[j][i][k + 2])) * 0.3 * uzf[j][i][k];

    for (j = 4; j <= _PB_N - 2; j++)
        for (i = 4; i <= _PB_N - 2; i++)
            for (k = 4; k <= _PB_N - 2; k++)
                athird[j][i][k] = a[j][i][k] + (al[j][i + 1][k] - al[j][i][k]) + (ab[j + 1][i][k] - ab[j][i][k]) + (af[j][i][k + 1] - af[j][i][k]);

                

#pragma endscop
}

int main(int argc, char **argv)
{
    /* Retrieve problem size. */
    int n = N;

    /* Variable declaration/allocation. */
    // POLYBENCH_3D_ARRAY_DECL(abB, DATA_TYPE, NX, NY, NK, nx, ny, n);
    POLYBENCH_3D_ARRAY_DECL(ab, DATA_TYPE, N, N, N, n, n, n);
    POLYBENCH_3D_ARRAY_DECL(al, DATA_TYPE, N, N, N, n, n, n);
    POLYBENCH_3D_ARRAY_DECL(af, DATA_TYPE, N, N, N, n, n, n);
    POLYBENCH_3D_ARRAY_DECL(a, DATA_TYPE, N, N, N, n, n, n);
    POLYBENCH_3D_ARRAY_DECL(athird, DATA_TYPE, N, N, N, n, n, n);
    POLYBENCH_3D_ARRAY_DECL(uyb, DATA_TYPE, N, N, N, n, n, n);
    POLYBENCH_3D_ARRAY_DECL(uxl, DATA_TYPE, N, N, N, n, n, n);
    POLYBENCH_3D_ARRAY_DECL(uzf, DATA_TYPE, N, N, N, n, n, n);

    /* Initialize array(s). */
    // init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

    /* Start timer. */
    polybench_start_instruments;

    /* Run kernel. */
    kernel_advect_3d(n, POLYBENCH_ARRAY(ab), POLYBENCH_ARRAY(al),
                     POLYBENCH_ARRAY(af), POLYBENCH_ARRAY(a),
                     POLYBENCH_ARRAY(athird), POLYBENCH_ARRAY(uyb),
                     POLYBENCH_ARRAY(uxl), POLYBENCH_ARRAY(uzf));

    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    // polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

    /* Be clean. */
    POLYBENCH_FREE_ARRAY(ab);
    POLYBENCH_FREE_ARRAY(al);
    POLYBENCH_FREE_ARRAY(af);
    POLYBENCH_FREE_ARRAY(a);
    POLYBENCH_FREE_ARRAY(athird);
    POLYBENCH_FREE_ARRAY(uyb);
    POLYBENCH_FREE_ARRAY(uxl);
    POLYBENCH_FREE_ARRAY(uzf);
    return 0;
}
