/*
 * Pluto: An automatic parallelizer and locality optimizer
 *
 * Copyright (C) 2007-2012 Uday Bondhugula
 *
 * This software is available under the MIT license. Please see LICENSE in the
 * top-level directory for details.
 *
 * This file is part of libpluto.
 *
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constraints.h"
#include "math_support.h"
#include "pluto/matrix.h"

/*
 * Allocated; not initialized.
 *
 * nrows and ncols initialized to allocated number of rows and cols
 */
PlutoMatrix *pluto_matrix_alloc(int alloc_nrows, int alloc_ncols,
                                PlutoContext *context) {
  assert(alloc_nrows >= 0);
  assert(alloc_ncols >= 0);

  PlutoMatrix *mat = (PlutoMatrix *)malloc(sizeof(PlutoMatrix));
  mat->context = context;
  mat->val = (int64_t **)malloc(PLMAX(alloc_nrows, 1) * sizeof(int64_t *));

  mat->alloc_nrows = PLMAX(alloc_nrows, 1);
  mat->alloc_ncols = PLMAX(alloc_ncols, 1);

  for (int i = 0; i < mat->alloc_nrows; i++) {
    mat->val[i] = (int64_t *)malloc(mat->alloc_ncols * sizeof(int64_t));
  }

  mat->nrows = alloc_nrows;
  mat->ncols = alloc_ncols;

  return mat;
}

void pluto_matrix_free(PlutoMatrix *mat) {
  int i;

  if (mat) {
    for (i = 0; i < mat->alloc_nrows; i++) {
      free(mat->val[i]);
    }

    free(mat->val);
    free(mat);
  }
}

/* Remove column <pos> from the matrix: pos starts from 0 */
void pluto_matrix_remove_col(PlutoMatrix *mat, int pos) {
  int i, j;

  assert(pos <= mat->ncols - 1);

  for (j = pos; j < mat->ncols - 1; j++) {
    for (i = 0; i < mat->nrows; i++) {
      mat->val[i][j] = mat->val[i][j + 1];
    }
  }
  mat->ncols--;
}

/* Remove row <pos> from the matrix: pos starts from 0 */
void pluto_matrix_remove_row(PlutoMatrix *mat, int pos) {
  int i, j;

  assert(pos <= mat->nrows - 1);

  for (i = pos; i < mat->nrows - 1; i++) {
    for (j = 0; j < mat->ncols; j++) {
      mat->val[i][j] = mat->val[i + 1][j];
    }
  }
  mat->nrows--;
}

/* Non-destructive resize */
void pluto_matrix_resize(PlutoMatrix *mat, int nrows, int ncols) {
  int i;

  int alloc_nrows = PLMAX(nrows, mat->alloc_nrows);
  int alloc_ncols = PLMAX(ncols, mat->alloc_ncols);

  mat->val = (int64_t **)realloc(mat->val, alloc_nrows * sizeof(int64_t *));

  for (i = mat->alloc_nrows; i < alloc_nrows; i++) {
    mat->val[i] = NULL;
  }

  for (i = 0; i < alloc_nrows; i++) {
    mat->val[i] =
        (int64_t *)realloc(mat->val[i], alloc_ncols * sizeof(int64_t));
  }

  mat->alloc_nrows = alloc_nrows;
  mat->alloc_ncols = alloc_ncols;

  mat->nrows = nrows;
  mat->ncols = ncols;
}

/* Add column to the matrix at <pos>: pos starts from 0;
 * New column is initialized to zero */
void pluto_matrix_add_col(PlutoMatrix *mat, int pos) {
  int i, j;

  assert(pos >= 0 && pos <= mat->ncols);

  if (mat->ncols == mat->alloc_ncols) {
    pluto_matrix_resize(mat, mat->nrows, mat->ncols + 1);
  } else {
    mat->ncols++;
  }

  for (j = mat->ncols - 2; j >= pos; j--) {
    for (i = 0; i < mat->nrows; i++) {
      mat->val[i][j + 1] = mat->val[i][j];
    }
  }

  /* Initialize to zero */
  for (i = 0; i < mat->nrows; i++) {
    mat->val[i][pos] = 0;
  }
}

/* Negate entire row; pos is 0-indexed */
void pluto_matrix_negate_row(PlutoMatrix *mat, int pos) {
  int j;

  for (j = 0; j < mat->ncols; j++) {
    mat->val[pos][j] = -mat->val[pos][j];
  }
}

void pluto_matrix_negate(PlutoMatrix *mat) {
  int r;
  for (r = 0; r < mat->nrows; r++) {
    pluto_matrix_negate_row(mat, r);
  }
}

/* Add rows of mat2 to mat1 */
void pluto_matrix_add(PlutoMatrix *mat1, const PlutoMatrix *mat2) {
  int i, j;

  assert(mat1->ncols == mat2->ncols);

  pluto_matrix_resize(mat1, mat1->nrows + mat2->nrows, mat1->ncols);

  for (i = mat1->nrows - mat2->nrows; i < mat1->nrows; i++) {
    for (j = 0; j < mat1->ncols; j++) {
      mat1->val[i][j] = mat2->val[i - (mat1->nrows - mat2->nrows)][j];
    }
  }
}

/* Add row to the matrix at <pos>: pos starts from 0; row is
 * initialized to zero */
void pluto_matrix_add_row(PlutoMatrix *mat, int pos) {
  int i, j;

  assert(mat != NULL);
  assert(pos <= mat->nrows);

  if (mat->nrows == mat->alloc_nrows) {
    pluto_matrix_resize(mat, mat->nrows + 1, mat->ncols);
  } else {
    mat->nrows++;
  }

  for (i = mat->nrows - 2; i >= pos; i--) {
    for (j = 0; j < mat->ncols; j++) {
      mat->val[i + 1][j] = mat->val[i][j];
    }
  }

  for (j = 0; j < mat->ncols; j++) {
    mat->val[pos][j] = 0;
  }
}

void pluto_matrix_interchange_rows(PlutoMatrix *mat, int r1, int r2) {
  /* int tmp, j; */

  for (int j = 0; j < mat->ncols; j++) {
    int64_t tmp = mat->val[r1][j];
    mat->val[r1][j] = mat->val[r2][j];
    mat->val[r2][j] = tmp;
  }
}

void pluto_matrix_interchange_cols(PlutoMatrix *mat, int c1, int c2) {
  int tmp, i;

  for (i = 0; i < mat->nrows; i++) {
    tmp = mat->val[i][c1];
    mat->val[i][c1] = mat->val[i][c2];
    mat->val[i][c2] = tmp;
  }
}

/* Move column from position c1 to c2 */
void pluto_matrix_move_col(PlutoMatrix *mat, int c1, int c2) {
  int j;

  if (c1 < c2) {
    for (j = c1; j < c2; j++) {
      pluto_matrix_interchange_cols(mat, j, j + 1);
    }
  } else {
    for (j = c1; j > c2; j--) {
      pluto_matrix_interchange_cols(mat, j, j - 1);
    }
  }
}

/* Return a duplicate of src */
PlutoMatrix *pluto_matrix_dup(const PlutoMatrix *src) {
  int i, j;

  assert(src != NULL);

  PlutoMatrix *dup =
      pluto_matrix_alloc(src->alloc_nrows, src->alloc_ncols, src->context);

  for (i = 0; i < src->nrows; i++) {
    for (j = 0; j < src->ncols; j++) {
      dup->val[i][j] = src->val[i][j];
    }
  }

  dup->nrows = src->nrows;
  dup->ncols = src->ncols;

  return dup;
}

/* Initialize matrix with val */
void pluto_matrix_set(PlutoMatrix *mat, int val) {
  int i, j;

  for (i = 0; i < mat->nrows; i++) {
    for (j = 0; j < mat->ncols; j++) {
      mat->val[i][j] = val;
    }
  }
}

/* Return an identity matrix of size: size x size */
PlutoMatrix *pluto_matrix_identity(int size, PlutoContext *context) {
  PlutoMatrix *mat = pluto_matrix_alloc(size, size, context);
  pluto_matrix_set(mat, 0);
  for (int i = 0; i < size; i++) {
    mat->val[i][i] = 1;
  }
  return mat;
}

/* Zero out a row */
void pluto_matrix_zero_row(PlutoMatrix *mat, int pos) {
  assert(pos >= 0 && pos <= mat->nrows - 1);

  for (int j = 0; j < mat->ncols; j++) {
    mat->val[pos][j] = 0;
  }
}

/* Zero out a column */
void pluto_matrix_zero_col(PlutoMatrix *mat, int pos) {
  int i;

  assert(pos >= 0 && pos <= mat->ncols - 1);

  for (i = 0; i < mat->nrows; i++) {
    mat->val[i][pos] = 0;
  }
}

void pluto_matrix_read(FILE *fp, const PlutoMatrix *mat) {
  int i, j;

  for (i = 0; i < mat->nrows; i++)
    for (j = 0; j < mat->ncols; j++)
      fscanf(fp, "%ld", &mat->val[i][j]);
}

PlutoMatrix *pluto_matrix_input(FILE *fp, PlutoContext *context) {
  int i, j, nrows, ncols;
  fscanf(fp, "%d %d", &nrows, &ncols);

  PlutoMatrix *mat = pluto_matrix_alloc(nrows, ncols, context);

  for (i = 0; i < mat->nrows; i++)
    for (j = 0; j < mat->ncols; j++)
      fscanf(fp, "%ld", &mat->val[i][j]);

  return mat;
}

void pluto_matrix_print(FILE *fp, const PlutoMatrix *mat) {
  int i, j;

  fprintf(fp, "%d %d\n", mat->nrows, mat->ncols);

  for (i = 0; i < mat->nrows; i++) {
    for (j = 0; j < mat->ncols; j++) {
      fprintf(fp, "%s%ld ", mat->val[i][j] >= 0 ? " " : "", mat->val[i][j]);
    }
    fprintf(fp, "\n");
  }
  // fprintf(fp, "\n");
}

/* Normalize row by its gcd */
void pluto_matrix_normalize_row(PlutoMatrix *mat, int pos) {
  int i, j, k;

  /* Normalize mat first */
  for (i = 0; i < mat->nrows; i++) {
    if (mat->val[i][0] == 0)
      continue;
    int rowgcd = llabs(mat->val[i][0]);
    for (j = 1; j < mat->ncols; j++) {
      if (mat->val[i][j] == 0)
        break;
      rowgcd = gcd(rowgcd, llabs(mat->val[i][j]));
    }
    if (i == mat->nrows) {
      if (rowgcd > 1) {
        for (k = 0; k < mat->ncols; k++) {
          mat->val[i][k] /= rowgcd;
        }
      }
    }
  }
}

/* pos: 0-indexed */
void gaussian_eliminate_var(PlutoMatrix *mat, int pos) {
  int r, r2, c;
  int factor1, factor2;

  for (r = 0; r < mat->nrows; r++) {
    if (mat->val[r][pos] != 0) {
      for (r2 = 0; r2 < mat->nrows; r2++) {
        if (r2 == r)
          continue;
        if (mat->val[r2][pos] != 0) {
          factor1 = lcm(llabs(mat->val[r][pos]), llabs(mat->val[r2][pos])) /
                    mat->val[r2][pos];
          factor2 = lcm(llabs(mat->val[r][pos]), llabs(mat->val[r2][pos])) /
                    mat->val[r][pos];
          for (c = 0; c < mat->ncols; c++) {
            mat->val[r2][c] =
                mat->val[r2][c] * factor1 - mat->val[r][c] * factor2;
          }
        }
      }
      pluto_matrix_remove_row(mat, r);
      break;
    }
  }

  pluto_matrix_remove_col(mat, pos);
}

/* Eliminate variables from start to end (inclusive); start is 0-indexed
 */
void gaussian_eliminate(PlutoMatrix *mat, int start, int num_elim) {
  int i;

  for (i = 0; i < num_elim; i++) {
    gaussian_eliminate_var(mat, start);
  }
}

int64_t lcm(int64_t a, int64_t b) {
  if (a * b == 0)
    return 0;
  return (a * b) / gcd(a, b);
}

/* Assuming both args are not zero */
int64_t gcd(int64_t a, int64_t b) {
  a = llabs(a);
  b = llabs(b);

  /* If at least one of them is zero */
  if (a * b == 0)
    return a + b;

  if (a == b)
    return a;

  return ((a > b) ? gcd(a % b, b) : gcd(a, b % a));
}

int64_t *min_lexical(int64_t *a, int64_t *b, int64_t num) {
  int i;

  for (i = 0; i < num; i++) {
    if (a[i] > b[i])
      return b;
    else if (a[i] < b[i])
      return a;
  }

  /* both are equal */
  return a;
}

/* Free returned string with free */
char *concat(const char *prefix, const char *suffix) {
  char *concat = (char *)malloc(strlen(prefix) + strlen(suffix) + 1);
  sprintf(concat, "%s%s", prefix, suffix);
  return concat;
}

PlutoMatrix *pluto_matrix_product(const PlutoMatrix *mat1,
                                  const PlutoMatrix *mat2) {
  assert(mat1->ncols == mat2->nrows);

  int i, j, k;

  PlutoMatrix *mat3 =
      pluto_matrix_alloc(mat1->nrows, mat2->ncols, mat1->context);

  for (i = 0; i < mat1->nrows; i++) {
    for (j = 0; j < mat2->ncols; j++) {
      mat3->val[i][j] = 0;
      for (k = 0; k < mat1->ncols; k++) {
        mat3->val[i][j] += mat1->val[i][k] * mat2->val[k][j];
      }
    }
  }
  return mat3;
}

/* Converts matrix to row-echelon form in-place */
PlutoMatrix *pluto_matrix_to_row_echelon(PlutoMatrix *mat) {
  int i, j, k, r, _lcm, factor1;

  r = 0;
  for (i = 0; i < PLMIN(mat->ncols, mat->nrows); i++) {
    if (mat->val[r][i] == 0) {
      for (k = r + 1; k < mat->nrows; k++) {
        if (mat->val[k][i] != 0)
          break;
      }
      if (k < mat->nrows) {
        pluto_matrix_interchange_rows(mat, r, k);
      }
    }
    if (mat->val[r][i] != 0) {
      for (k = r + 1; k < mat->nrows; k++) {
        if (mat->val[k][i] == 0)
          continue;
        _lcm = lcm(mat->val[k][i], mat->val[r][i]);
        factor1 = _lcm / mat->val[k][i];
        for (j = i; j < mat->ncols; j++) {
          mat->val[k][j] = mat->val[k][j] * factor1 -
                           mat->val[r][j] * (_lcm / mat->val[r][i]);
        }
      }
      r++;
    }
  }

  return mat;
}

/* Rank of the matrix */
unsigned pluto_matrix_get_rank(const PlutoMatrix *mat) {
  unsigned rank;

  PlutoMatrix *re = pluto_matrix_to_row_echelon(pluto_matrix_dup(mat));

  unsigned null = 0;
  for (unsigned i = 0; i < re->nrows; i++) {
    int sum = 0;
    for (unsigned j = 0; j < re->ncols; j++) {
      sum += llabs(re->val[i][j]);
    }
    if (sum == 0)
      null++;
  }
  rank = re->nrows - null;
  pluto_matrix_free(re);
  return rank;
}

void pluto_matrix_swap_rows(PlutoMatrix *mat, int r1, int r2) {
  int64_t tmp;
  int j;

  for (j = 0; j < mat->ncols; j++) {
    tmp = mat->val[r2][j];
    mat->val[r2][j] = mat->val[r1][j];
    mat->val[r1][j] = tmp;
  }
}

/* Reverse the order of rows in the matrix */
void pluto_matrix_reverse_rows(PlutoMatrix *mat) {
  int i;

  for (i = 0; i < mat->nrows / 2; i++) {
    pluto_matrix_swap_rows(mat, i, mat->nrows - 1 - i);
  }
}

/*
 * Pretty prints a one-dimensional affine function
 * ndims: number of variables
 * func should have ndims+1 elements (affine function)
 * vars: names of the ndims variables; if NULL, x0, x1, ... are used
 */
void pluto_affine_function_print(FILE *fp, int64_t *func, int ndims,
                                 const char **vars) {
  char *var[ndims];
  int j;

  for (j = 0; j < ndims; j++) {
    if (vars && vars[j]) {
      var[j] = strdup(vars[j]);
    } else {
      var[j] = (char *)malloc(5);
      sprintf(var[j], "x%d", j + 1);
    }
  }

  int first = 0;
  for (j = 0; j < ndims; j++) {
    if (func[j] == 1) {
      if (first)
        fprintf(fp, "+");
      fprintf(fp, "%s", var[j]);
    } else if (func[j] == -1) {
      fprintf(fp, "-%s", var[j]);
    } else if (func[j] != 0) {
      if (func[j] > 0) {
        fprintf(fp, "%s%ld%s", first ? "+" : "", func[j], var[j]);
      } else {
        fprintf(fp, "%ld%s", func[j], var[j]);
      }
    }
    if (func[j] != 0)
      first = 1;
  }
  /* Constant part */
  if (func[ndims] >= 1) {
    if (first)
      fprintf(fp, "+");
    fprintf(fp, "%ld", func[ndims]);
  } else if (func[ndims] <= -1) {
    fprintf(fp, "%ld", func[ndims]);
  } else {
    /* 0 */
    if (!first)
      fprintf(fp, "0");
  }

  for (j = 0; j < ndims; j++) {
    free(var[j]);
  }
}

/* Returned string should be freed with malloc */
char *pluto_affine_function_sprint(int64_t *func, int ndims,
                                   const char **vars) {
  char *var[ndims], *out;
  int j, n;

  /* max 5 chars for var, 3 for coefficient + 1 if ndims is 0 + 1 null char */
  n = 9 * ndims + 1 + 1;
  out = (char *)malloc(n);
  *out = '\0';

  for (j = 0; j < ndims; j++) {
    if (vars && vars[j]) {
      var[j] = strdup(vars[j]);
    } else {
      var[j] = (char *)malloc(5);
      sprintf(var[j], "x%d", j + 1);
    }
  }

  int first = 0;
  for (j = 0; j < ndims; j++) {
    if (func[j] == 1) {
      if (first)
        strcat(out, "+");
      snprintf(out + strlen(out), 5, "%s", var[j]);
    } else if (func[j] == -1) {
      snprintf(out + strlen(out), 6, "-%s", var[j]);
    } else if (func[j] != 0) {
      if (func[j] >= 1) {
        snprintf(out + strlen(out), n - strlen(out), "%s%ld%s",
                 first ? "+" : "", func[j], var[j]);
      } else {
        snprintf(out + strlen(out), n - strlen(out), "%ld%s", func[j], var[j]);
      }
    }
    if (func[j] != 0)
      first = 1;
  }
  /* Constant part */
  if (func[ndims] >= 1) {
    if (first)
      strcat(out, "+");
    snprintf(out + strlen(out), 3, "%ld", func[ndims]);
  } else if (func[ndims] <= -1) {
    snprintf(out + strlen(out), 3, "%ld", func[ndims]);
  } else {
    /* 0 */
    if (!first)
      strcat(out, "0");
  }

  for (j = 0; j < ndims; j++) {
    free(var[j]);
  }

  return out;
}

/*
 * Is row r1 of mat1 parallel to row r2 of mat2
 */
int pluto_vector_is_parallel(PlutoMatrix *mat1, int r1, PlutoMatrix *mat2,
                             int r2) {
  int num, den, j;

  assert(mat1->ncols == mat2->ncols);

  num = 0;
  den = 0;

  for (j = 0; j < mat1->ncols; j++) {
    if (mat1->val[r1][j] == 0 && mat2->val[r2][j] == 0)
      continue;
    if (mat1->val[r1][j] != 0 && mat2->val[r2][j] == 0)
      return 0;
    if (mat1->val[r1][j] == 0 && mat2->val[r2][j] != 0)
      return 0;

    /* num and den are always non-zero */
    if (num == 0) {
      /* first time */
      num = mat1->val[r1][j];
      den = mat2->val[r2][j];
    } else {
      if (num * mat2->val[r2][j] != den * mat1->val[r1][j])
        return 0;
    }
  }

  return 1;
}

int pluto_vector_is_normal(PlutoMatrix *mat1, int r1, PlutoMatrix *mat2,
                           int r2) {
  int j, dot;

  assert(mat1->ncols == mat2->ncols);

  dot = 0;
  for (j = 0; j < mat1->ncols; j++) {
    dot += mat1->val[r1][j] * mat2->val[r2][j];
  }

  return (dot == 0);
}

/* Convert from mpz to signed long long */
void mpz_set_sll(mpz_t n, long long sll) {
  /* n = (int)sll >> 32 */
  mpz_set_si(n, (int)(sll >> 32));
  /* n <<= 32 */
  mpz_mul_2exp(n, n, 32);
  /* n += (unsigned int)sll */
  mpz_add_ui(n, n, (unsigned int)sll);
}

/* Convert from mpz to unsigned long long */
void mpz_set_ull(mpz_t n, unsigned long long ull) {
  /* n = (unsigned int)(ull >> 32) */
  mpz_set_ui(n, (unsigned int)(ull >> 32));
  /* n <<= 32 */
  mpz_mul_2exp(n, n, 32);
  /* n += (unsigned int)ull */
  mpz_add_ui(n, n, (unsigned int)ull);
}

/// Returns true if the two input matrices are equal.
bool are_pluto_matrices_equal(PlutoMatrix *mat1, PlutoMatrix *mat2) {
  if ((mat1->nrows != mat2->nrows) || (mat1->ncols != mat2->ncols))
    return false;
  for (unsigned i = 0; i < mat1->nrows; i++) {
    for (unsigned j = 0; j < mat1->ncols; j++) {
      if (mat1->val[i][j] != mat2->val[i][j])
        return false;
    }
  }
  return true;
}
