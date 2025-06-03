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
 * Ploop is a loop dimension in the scattering tree. When the final
 * AST is generated, a single Ploop can get separated into multiple ones.
 *
 */
#include <assert.h>
#include <stdbool.h>
#include <string.h>

#include "math_support.h"
#include "pluto/internal/pluto.h"
#include "pluto/matrix.h"
#include "pluto/pluto.h"
#include "program.h"

/// Number of registers defined in the ISA. This is used in heuristics that
/// determine whether the unroll jam of a loop is profitable.
#define NUM_TOTAL_REGISTERS 32
Ploop *pluto_loop_alloc() {
  Ploop *loop = (Ploop *)malloc(sizeof(Ploop));
  loop->nstmts = 0;
  loop->depth = -1;
  loop->stmts = NULL;
  return loop;
}

void pluto_loop_print(const Ploop *loop) {
  fprintf(stderr, "t%d {loop with stmts: ", loop->depth + 1);
  for (unsigned i = 0; i < loop->nstmts; i++) {
    fprintf(stderr, "S%d, ", loop->stmts[i]->id + 1);
  }
  fprintf(stderr, "}\n");
}

void pluto_loops_print(Ploop **loops, int num) {
  for (int i = 0; i < num; i++) {
    pluto_loop_print(loops[i]);
  }
}

Ploop *pluto_loop_dup(Ploop *l) {
  Ploop *loop;
  loop = pluto_loop_alloc();
  loop->depth = l->depth;
  loop->nstmts = l->nstmts;
  loop->stmts = (Stmt **)malloc(loop->nstmts * sizeof(Stmt *));
  memcpy(loop->stmts, l->stmts, l->nstmts * sizeof(Stmt *));
  return loop;
}

void pluto_loop_free(Ploop *loop) {
  if (loop->stmts != NULL) {
    free(loop->stmts);
  }
  free(loop);
}

void pluto_loops_free(Ploop **loops, int nloops) {
  int i;
  for (i = 0; i < nloops; i++) {
    pluto_loop_free(loops[i]);
  }
  free(loops);
}

/*
 * Concatenates two arrays of loops and places them in the first argument;
 * pointers are copied - not the structures; destination is resized if
 * necessary */
Ploop **pluto_loops_cat(Ploop **dest, int num1, Ploop **src, int num2) {
  assert(num1 >= 0 && num2 >= 0);
  if (num1 == 0 && num2 == 0) {
    return dest;
  }
  dest = (Ploop **)realloc(dest, (num1 + num2) * sizeof(Ploop *));
  memcpy(dest + num1, src, num2 * sizeof(Ploop *));
  return dest;
}

Ploop **pluto_get_loops_immediately_inner(Ploop *ploop, const PlutoProg *prog,
                                          unsigned *num) {
  unsigned ni;

  Ploop **loops = pluto_get_loops_under(ploop->stmts, ploop->nstmts,
                                        ploop->depth, prog, &ni);

  *num = 0;

  Ploop **imloops = NULL;

  for (unsigned i = 0; i < ni; i++) {
    if (loops[i]->depth == ploop->depth + 1) {
      Ploop *loop = pluto_loop_dup(loops[i]);
      imloops = pluto_loops_cat(imloops, (*num)++, &loop, 1);
    }
  }

  pluto_loops_free(loops, ni);

  return imloops;
}

/* Get loops at this depth under the scattering tree that contain only and all
 * of the statements in 'stmts'
 * If all 'stmts' aren't fused up to depth-1, no such loops exist by
 * definition
 */
Ploop **pluto_get_loops_under(Stmt **stmts, unsigned nstmts, unsigned depth,
                              const PlutoProg *prog, unsigned *num) {
  Ploop **all_loops = NULL;
  bool loop;

  if (depth >= prog->num_hyperplanes) {
    *num = 0;
    return NULL;
  }

  unsigned i;
  for (i = 0; i < nstmts; i++) {
    if (stmts[i]->type != ORIG)
      continue;
    if (pluto_is_hyperplane_loop(stmts[i], depth))
      break;
  }

  loop = (i < nstmts);

  if (loop) {
    /* Even if one of the statements is a loop at this level,
     * all of the statements are treated as being fused with it;
     * check any limitations of this later */
    unsigned num_inner, loop_nstmts, i;
    Ploop **inner_loops, *this_loop;
    this_loop = pluto_loop_alloc();
    this_loop->stmts = (Stmt **)malloc(nstmts * sizeof(Stmt *));

    loop_nstmts = 0;
    for (i = 0; i < nstmts; i++) {
      if (!pluto_is_hyperplane_scalar(stmts[i], depth)) {
        memcpy(this_loop->stmts + loop_nstmts, stmts + i, sizeof(Stmt *));
        loop_nstmts++;
      }
    }
    this_loop->nstmts = loop_nstmts;
    this_loop->depth = depth;
    /* Even with H_LOOP, some statements can have a scalar dimension - at
     * least one of the statements is a real loop (doesn't happen often
     * though */
    /* TODO: What if the hyperplane was 0 for S1, 1 for S2, i for S3 */
    /* The filter (first arg) for call below is not changed since if a
     * stmt has a scalar dimension here, it may still be fused with
     * other ones which do not have a scalar dimension here */
    inner_loops =
        pluto_get_loops_under(stmts, nstmts, depth + 1, prog, &num_inner);
    if (loop_nstmts >= 1) {
      inner_loops = pluto_loops_cat(inner_loops, num_inner, &this_loop, 1);
      *num = num_inner + 1;
    } else {
      pluto_loop_free(this_loop);
      *num = num_inner;
    }
    all_loops = inner_loops;
  } else {
    /* All statements have a scalar dimension at this depth */
    /* Regroup stmts */
    Stmt ***groups;
    int i, j, num_grps;
    int *num_stmts_per_grp;

    num_grps = 0;
    num_stmts_per_grp = NULL;
    groups = NULL;

    for (i = 0; i < nstmts; i++) {
      Stmt *stmt = stmts[i];
      for (j = 0; j < num_grps; j++) {
        if (stmt->trans->val[depth][stmt->trans->ncols - 1] ==
            groups[j][0]->trans->val[depth][groups[j][0]->trans->ncols - 1]) {
          /* Add to end of array */
          groups[j] = (Stmt **)realloc(groups[j], (num_stmts_per_grp[j] + 1) *
                                                      sizeof(Stmt *));
          groups[j][num_stmts_per_grp[j]] = stmt;
          num_stmts_per_grp[j]++;
          break;
        }
      }
      if (j == num_grps) {
        /* New group */
        groups = (Stmt ***)realloc(groups, (num_grps + 1) * sizeof(Stmt **));
        groups[num_grps] = (Stmt **)malloc(sizeof(Stmt *));
        groups[num_grps][0] = stmt;

        num_stmts_per_grp =
            (int *)realloc(num_stmts_per_grp, (num_grps + 1) * sizeof(int));
        num_stmts_per_grp[num_grps] = 1;
        num_grps++;
      }
    }

    int nloops = 0;
    for (i = 0; i < num_grps; i++) {
      unsigned num_inner = 0;
      Ploop **inner_loops;
      inner_loops = pluto_get_loops_under(groups[i], num_stmts_per_grp[i],
                                          depth + 1, prog, &num_inner);
      all_loops = pluto_loops_cat(all_loops, nloops, inner_loops, num_inner);
      nloops += num_inner;
      free(inner_loops);
    }
    *num = nloops;

    for (i = 0; i < num_grps; i++) {
      free(groups[i]);
    }

    free(num_stmts_per_grp);
    free(groups);
  }

  return all_loops;
}

Ploop **pluto_get_all_loops(const PlutoProg *prog, unsigned *num) {
  Ploop **loops;
  loops = pluto_get_loops_under(prog->stmts, prog->nstmts, 0, prog, num);
  return loops;
}

/*
 * Whether the loop has a non-zero dependence component for an already
 * satisfied dependence
 */
int pluto_loop_has_satisfied_dep_with_component(const PlutoProg *prog,
                                                const Ploop *loop) {
  unsigned i;
  /* All statements under a parallel loop should be of type orig */
  for (i = 0; i < loop->nstmts; i++) {
    if ((loop->stmts[i]->type != ORIG))
      break;
  }
  if (i < loop->nstmts) {
    return 1;
  }

  int retval = 0;

  for (i = 0; i < prog->ndeps; i++) {
    Dep *dep = prog->deps[i];
    if (IS_RAR(dep->type))
      continue;
    assert(dep->satvec != NULL);
    if (pluto_stmt_is_member_of(prog->stmts[dep->src]->id, loop->stmts,
                                loop->nstmts) &&
        pluto_stmt_is_member_of(prog->stmts[dep->dest]->id, loop->stmts,
                                loop->nstmts) &&
        dep->satisfaction_level < loop->depth &&
        (dep->dirvec[loop->depth] == DEP_STAR ||
         dep->dirvec[loop->depth] == DEP_PLUS ||
         dep->dirvec[loop->depth] == DEP_MINUS)) {
      break;
    }
  }

  if (i < prog->ndeps)
    retval = 1;

  return retval;
}

int pluto_loop_is_parallel(const PlutoProg *prog, Ploop *loop) {
  int i;

  /* All statements under a parallel loop should be of type orig */
  for (i = 0; i < loop->nstmts; i++) {
    if (loop->stmts[i]->type != ORIG)
      break;
  }
  if (i < loop->nstmts) {
    return 0;
  }

  if (prog->hProps[loop->depth].dep_prop == PARALLEL) {
    return 1;
  }

  int parallel = 1;

  for (i = 0; i < prog->ndeps; i++) {
    Dep *dep = prog->deps[i];
    if (IS_RAR(dep->type))
      continue;
    assert(dep->satvec != NULL);
    if (pluto_stmt_is_member_of(prog->stmts[dep->src]->id, loop->stmts,
                                loop->nstmts) &&
        pluto_stmt_is_member_of(prog->stmts[dep->dest]->id, loop->stmts,
                                loop->nstmts)) {
      if (dep->satvec[loop->depth]) {
        parallel = 0;
        break;
      }
    }
  }

  return parallel;
}

/*
 * Whether all statements instances of 'stmt' across different iterations of
 * loop can be run in parallel; only used in a special context; it doesn't
 * mean whether the loop is itself parallel in any way.
 */
int pluto_loop_is_parallel_for_stmt(const PlutoProg *prog, const Ploop *loop,
                                    const Stmt *stmt) {
  int i;

  /* All statements under a parallel loop should be of type orig */
  for (i = 0; i < loop->nstmts; i++) {
    if ((loop->stmts[i]->type != ORIG))
      break;
  }
  if (i < loop->nstmts) {
    return 0;
  }

  if (prog->hProps[loop->depth].dep_prop == PARALLEL) {
    return 1;
  }

  int parallel = 1;

  for (i = 0; i < prog->ndeps; i++) {
    Dep *dep = prog->deps[i];
    if (IS_RAR(dep->type))
      continue;
    assert(dep->satvec != NULL);
    if (prog->stmts[dep->src]->id == stmt->id &&
        prog->stmts[dep->dest]->id == stmt->id && dep->satvec[loop->depth]) {
      parallel = 0;
      break;
    }
  }

  return parallel;
}

/*
 * Does depth and anything below satisfy an inter-stmt dependence?
 */
int pluto_satisfies_inter_stmt_dep(const PlutoProg *prog, const Ploop *loop,
                                   int depth) {
  int i;

  /* All statements under a parallel loop should be of type orig */
  for (i = 0; i < loop->nstmts; i++) {
    if ((loop->stmts[i]->type != ORIG))
      break;
  }
  if (i < loop->nstmts) {
    /* conservative */
    return 1;
  }

  int satisfies = 0;

  for (i = 0; i < prog->ndeps; i++) {
    Dep *dep = prog->deps[i];
    if (IS_RAR(dep->type))
      continue;
    assert(dep->satvec != NULL);

    if (dep->src == dep->dest ||
        !pluto_stmt_is_member_of(dep->src, loop->stmts, loop->nstmts) ||
        !pluto_stmt_is_member_of(dep->dest, loop->stmts, loop->nstmts)) {
      continue;
    }

    for (unsigned d = depth; d < prog->num_hyperplanes; d++) {
      if (dep->satvec[d]) {
        satisfies = 1;
        break;
      }
    }
    if (satisfies)
      break;
  }

  return satisfies;
}

/* Is loop1 dominated by loop2 */
int is_loop_dominated(Ploop *loop1, Ploop *loop2, const PlutoProg *prog) {
  assert(loop1->nstmts >= 1);
  assert(loop2->nstmts >= 1);

  if (loop2->depth >= loop1->depth)
    return 0;

  Stmt *stmt1 = loop1->stmts[0];
  Stmt *stmt2 = loop2->stmts[0];

  for (unsigned i = 0; i <= loop2->depth; i++) {
    if (pluto_is_hyperplane_scalar(stmt1, i) &&
        pluto_is_hyperplane_scalar(stmt2, i)) {
      if (stmt1->trans->val[i][stmt1->trans->ncols - 1] !=
          stmt2->trans->val[i][stmt2->trans->ncols - 1]) {
        return 0;
      }
    }
  }
  return 1;
}

/* For sorting loops in increasing order of their depths */
int pluto_loop_compar(const void *_l1, const void *_l2) {
  Ploop *l1 = *(Ploop **)_l1;
  Ploop *l2 = *(Ploop **)_l2;

  if (l1->depth < l2->depth) {
    return -1;
  }

  if (l1->depth == l2->depth) {
    return 0;
  }

  return 1;
}

unsigned get_num_accesses(Ploop *loop) {
  /* All statements under the loop, all accesses for the statement */
  unsigned ns = 0;
  for (unsigned i = 0; i < loop->nstmts; i++) {
    ns += loop->stmts[i]->nreads + loop->stmts[i]->nwrites;
  }
  return ns;
}

/// Is the access given by acc an invariant at a given depth. If yes, then the
/// access has temporal reuse along the hyperplane at this depth.
static bool is_invariant(Stmt *stmt, PlutoAccess *acc, int depth) {
  int *divs;
  PlutoMatrix *newacc = pluto_get_new_access_func(acc->mat, stmt, &divs);
  assert(depth <= (int)newacc->ncols - 1);
  unsigned i;
  for (i = 0; i < newacc->nrows; i++) {
    if (newacc->val[i][depth] != 0)
      break;
  }
  bool is_invariant = (i == newacc->nrows);
  pluto_matrix_free(newacc);
  free(divs);
  return is_invariant;
}

unsigned get_num_invariant_accesses(Ploop *loop) {
  /* All statements under the loop, all accesses for the statement */
  unsigned ni = 0;
  for (unsigned i = 0; i < loop->nstmts; i++) {
    Stmt *stmt = loop->stmts[i];
    for (int j = 0; j < stmt->nreads; j++) {
      ni += is_invariant(stmt, stmt->reads[j], loop->depth);
    }
    for (int j = 0; j < stmt->nwrites; j++) {
      ni += is_invariant(stmt, stmt->writes[j], loop->depth);
    }
  }
  return ni;
}

/// Returns the maximum dimensionality of the set of input statements.
static unsigned get_max_dim(Stmt **stmts, unsigned nstmts) {
  unsigned max_dim = stmts[0]->dim;
  for (unsigned j = 1; j < nstmts; j++) {
    Stmt *stmt = stmts[j];
    if (stmt->dim > max_dim)
      max_dim = stmt->dim;
  }
  return max_dim;
}

unsigned get_max_num_innermost_invariant_accesses(Ploop *loop,
                                                  const PlutoProg *prog) {
  unsigned num_inner_loops;
  Ploop **iloops = pluto_get_loops_under(loop->stmts, loop->nstmts, loop->depth,
                                         prog, &num_inner_loops);
  unsigned max_invariant_accesses = 0;
  Stmt **stmts = (Stmt **)malloc(loop->nstmts * sizeof(Stmt *));
  for (unsigned i = 0; i < num_inner_loops; i++) {
    if (!pluto_loop_is_innermost(iloops[i], prog))
      continue;
    unsigned max_dim = get_max_dim(iloops[i]->stmts, iloops[i]->nstmts);
    unsigned nstmts = 0;
    for (unsigned j = 0; j < iloops[i]->nstmts; j++) {
      Stmt *stmt = iloops[i]->stmts[j];
      if (stmt->dim != max_dim)
        continue;
      stmts[nstmts++] = stmt;
    }
    unsigned unique_invariant_accesses =
        get_num_invariant_accesses_in_stmts(stmts, nstmts, loop->depth, prog);
    if (unique_invariant_accesses > max_invariant_accesses)
      max_invariant_accesses = unique_invariant_accesses;
  }
  free(stmts);
  pluto_loops_free(iloops, num_inner_loops);
  return max_invariant_accesses;
}

/// Returns the maximum number of accesses at the innermost level.
static unsigned get_max_num_innermost_accesses(Ploop *loop,
                                               const PlutoProg *prog) {
  unsigned num_inner_loops;
  Ploop **iloops = pluto_get_loops_under(loop->stmts, loop->nstmts, loop->depth,
                                         prog, &num_inner_loops);

  unsigned max_unique_accesses = 0;
  Stmt **stmts = (Stmt **)malloc(loop->nstmts * sizeof(Stmt *));
  for (unsigned i = 0; i < num_inner_loops; i++) {
    if (!pluto_loop_is_innermost(iloops[i], prog))
      continue;

    unsigned max_dim = get_max_dim(iloops[i]->stmts, iloops[i]->nstmts);
    unsigned nstmts = 0;
    for (unsigned j = 0; j < iloops[i]->nstmts; j++) {
      Stmt *stmt = iloops[i]->stmts[j];
      if (stmt->dim != max_dim)
        continue;
      stmts[nstmts++] = stmt;
    }
    unsigned unique_accesses =
        get_num_unique_accesses_in_stmts(stmts, nstmts, prog);
    if (unique_accesses > max_unique_accesses)
      max_unique_accesses = unique_accesses;
  }
  free(stmts);
  pluto_loops_free(iloops, num_inner_loops);
  return max_unique_accesses;
}

/// Cost function returns true if unrolling jamming the loop will utilize less
/// number of registers than the total number of registers specified in the ISA
/// of the processor. We assume that the total number of registers that is
/// available is 32. This heuristic has to be tuned further.
bool is_unroll_jam_profitable(Ploop *loop, const PlutoProg *prog) {
  PlutoContext *context = prog->context;
  PlutoOptions *options = context->options;
  unsigned t = get_max_num_innermost_invariant_accesses(loop, prog);
  IF_DEBUG(printf("Number of accesses with temporal reuse: %d\n", t););
  // If there is no temporal reuse, then unroll jam isnt profitable.
  if (t == 0)
    return false;
  unsigned a = get_max_num_innermost_accesses(loop, prog);
  IF_DEBUG(pluto_loop_print(loop););
  IF_DEBUG(
      printf("Total number of unique accesses at innermost level %d\n", a););
  unsigned unroll_factor = options->ufactor;
  int num_reg_required = a * unroll_factor - (t * (unroll_factor - 1));
  int cost = NUM_TOTAL_REGISTERS - num_reg_required;
  IF_DEBUG(pluto_loop_print(loop););
  IF_DEBUG(printf("Cost for loop: %d\n", cost););
  if (cost >= 0)
    return true;
  return false;
}

/// Returns a list of intra tile loops that are candidates for unroll jam.
Ploop **pluto_get_unroll_jam_loops(const PlutoProg *prog,
                                   unsigned *num_ujloops) {
  unsigned num_ibands;
  Band **ibands = pluto_get_innermost_permutable_bands(prog, 1, &num_ibands);
  unsigned num = 0;
  Ploop **ujloops = NULL;
  unsigned nloops = 0;
  for (unsigned j = 0; j < num_ibands; j++) {
    Ploop *loop = ibands[j]->loop;
    Ploop **loops = pluto_get_loops_under(loop->stmts, loop->nstmts,
                                          loop->depth, prog, &num);
    for (unsigned i = 0; i < num; i++) {
      /* Do not unroll jam a tile space loop. */
      if (is_tile_space_loop(loops[i], prog))
        continue;
      /* Do not unroll jam the innermost loop. */
      if (pluto_loop_is_innermost(loops[i], prog))
        continue;
      /* Do not unroll jam if unroll jamming is not profitable. Refer function
       * defintion for the cost function. */
      if (!is_unroll_jam_profitable(loops[i], prog))
        continue;
      ujloops = (Ploop **)realloc(ujloops, (nloops + 1) * sizeof(Ploop *));
      ujloops[nloops++] = pluto_loop_dup(loops[i]);
    }
    pluto_loops_free(loops, num);
  }
  *num_ujloops = nloops;
  pluto_bands_free(ibands, num_ibands);
  return ujloops;
}

/* Get all parallel loops */
Ploop **pluto_get_parallel_loops(const PlutoProg *prog, unsigned *nploops) {
  Ploop **loops, **ploops;
  unsigned num = 0, i;

  ploops = NULL;
  loops = pluto_get_all_loops(prog, &num);

  // pluto_loops_print(loops, num);

  *nploops = 0;
  for (i = 0; i < num; i++) {
    if (pluto_loop_is_parallel(prog, loops[i])) {
      ploops = (Ploop **)realloc(ploops, (*nploops + 1) * sizeof(Ploop *));
      ploops[*nploops] = pluto_loop_dup(loops[i]);
      (*nploops)++;
    }
  }

  pluto_loops_free(loops, num);

  return ploops;
}

/* List of parallel loops such that no loop dominates another in the list */
Ploop **pluto_get_dom_parallel_loops(const PlutoProg *prog,
                                     unsigned *ndploops) {
  Ploop **loops, **dom_loops;
  unsigned j, ndomloops, nploops;

  loops = pluto_get_parallel_loops(prog, &nploops);

  dom_loops = NULL;
  ndomloops = 0;
  for (unsigned i = 0; i < nploops; i++) {
    for (j = 0; j < nploops; j++) {
      if (is_loop_dominated(loops[i], loops[j], prog))
        break;
    }
    if (j == nploops) {
      Ploop *loop = pluto_loop_dup(loops[i]);
      dom_loops = pluto_loops_cat(dom_loops, ndomloops++, &loop, 1);
    }
  }
  *ndploops = ndomloops;
  pluto_loops_free(loops, nploops);

  return dom_loops;
}

/* Returns a list of outermost loops */
Ploop **pluto_get_outermost_loops(const PlutoProg *prog, int *ndploops) {
  Ploop **loops, **dom_loops;
  int i, j;
  unsigned ndomloops, nploops;

  loops = pluto_get_all_loops(prog, &nploops);

  dom_loops = NULL;
  ndomloops = 0;
  for (i = 0; i < nploops; i++) {
    for (j = 0; j < nploops; j++) {
      if (is_loop_dominated(loops[i], loops[j], prog))
        break;
    }
    if (j == nploops) {
      dom_loops = pluto_loops_cat(dom_loops, ndomloops++, &loops[i], 1);
    }
  }
  *ndploops = ndomloops;

  return dom_loops;
}

/* Band support */

Band *pluto_band_alloc(Ploop *loop, int width) {
  Band *band = (Band *)malloc(sizeof(Band));
  band->width = width;
  band->loop = pluto_loop_dup(loop);
  band->children = NULL;
  return band;
}

void pluto_band_print(const Band *band) {
  int i;
  printf("(");
  for (i = band->loop->depth; i < band->loop->depth + band->width; i++) {
    printf("t%d, ", i + 1);
  }
  printf(") with stmts {");
  for (i = 0; i < band->loop->nstmts; i++) {
    printf("S%d, ", band->loop->stmts[i]->id + 1);
  }
  printf("}\n");
}

void pluto_bands_print(Band **bands, int num) {
  int i;
  if (num == 0)
    printf("0 bands\n");
  for (i = 0; i < num; i++) {
    pluto_band_print(bands[i]);
  }
}

Band *pluto_band_dup(Band *b) { return pluto_band_alloc(b->loop, b->width); }

/* Concatenates both and puts them in bands1; pointers are copied - not
 * the structures */
Band **pluto_bands_cat(Band ***dest, int num1, Band **src, int num2) {
  if (num1 == 0 && num2 == 0)
    return *dest;
  *dest = (Band **)realloc(*dest, (num1 + num2) * sizeof(Band *));
  memcpy((*dest) + num1, src, num2 * sizeof(Band *));
  return *dest;
}

/* Is band1 dominated by band2; a band is dominated by another if the latter
 * either
 * completely includes the former or the former is in the subtree rooted at the
 * parent */
static int is_band_dominated(Band *band1, Band *band2) {
  int is_subset =
      pluto_stmt_is_subset_of(band1->loop->stmts, band1->loop->nstmts,
                              band2->loop->stmts, band2->loop->nstmts);

  if (is_subset && band1->loop->depth > band2->loop->depth)
    return 1;

  if (is_subset && band1->loop->depth == band2->loop->depth &&
      band1->width < band2->width)
    return 1;

  return 0;
}

void pluto_band_free(Band *band) {
  if (band) {
    pluto_loop_free(band->loop);
    free(band);
  }
}

void pluto_bands_free(Band **bands, unsigned nbands) {
  for (unsigned i = 0; i < nbands; i++) {
    pluto_band_free(bands[i]);
  }
  free(bands);
}

/// Returns true, if all statements under the loop have a scalar hyperplane at
/// depth 'depth'.
bool pluto_is_depth_scalar(Ploop *loop, int depth) {

  assert(depth >= loop->depth);

  for (int i = 0; i < loop->nstmts; i++) {
    if (!pluto_is_hyperplane_scalar(loop->stmts[i], depth))
      return false;
  }

  return true;
}

/* Returns a non-trivial permutable band starting from this loop; NULL
 * if the band is trivial (just the loop itself */
Band *pluto_get_permutable_band(Ploop *loop, const PlutoProg *prog) {
  int i, depth;

  depth = loop->depth;

  do {
    for (i = 0; i < prog->ndeps; i++) {
      Dep *dep = prog->deps[i];
      if (IS_RAR(dep->type))
        continue;
      assert(dep->satvec != NULL);
      /* Dependences satisfied outer to the band don't matter */
      if (dep->satisfaction_level < loop->depth)
        continue;
      /* Dependences not satisfied yet don't matter */
      if (!dep->satisfied)
        continue;
      /* Dependences satisfied in previous scalar dimensions in the band
       * don't count as well (i.e., they can have negative components) */
      if (pluto_is_depth_scalar(loop, dep->satisfaction_level))
        continue;
      /* Rest of the dependences need to have non-negative components */
      if (pluto_stmt_is_member_of(prog->stmts[dep->src]->id, loop->stmts,
                                  loop->nstmts) &&
          pluto_stmt_is_member_of(prog->stmts[dep->dest]->id, loop->stmts,
                                  loop->nstmts)) {
        if (dep->dirvec[depth] == DEP_STAR || dep->dirvec[depth] == DEP_MINUS)
          break;
      }
    }
    if (i < prog->ndeps)
      break;
    depth++;
  } while (depth < prog->num_hyperplanes);

  /* Peel off scalar dimensions from the end */
  while (pluto_is_depth_scalar(loop, depth - 1))
    depth--;

  /* depth will always be at least loop->depth + 1 at this stage */

  int width = depth - loop->depth;

  assert(width >= 1);

  if (width == 1) {
    /* Trivial (single loop in band) */
    return NULL;
  } else {
    return pluto_band_alloc(loop, width);
  }
}

/* Returns a parallel band starting from this loop; the band could end up
 * being just this loop itself
 * innermost_split_level: the first parallel loop in the band which
 * has distribution under it (loop->depth is the band is perfectly nested)
 */
Band *pluto_get_parallel_band(Ploop *loop, const PlutoProg *prog,
                              int *innermost_split_level) {
  int i, d, depth, width;

  assert(!pluto_is_depth_scalar(loop, loop->depth));

  depth = loop->depth;

  *innermost_split_level = loop->depth;

  do {
    for (i = 0; i < prog->ndeps; i++) {
      Dep *dep = prog->deps[i];
      if (IS_RAR(dep->type))
        continue;
      /* Dependences where both the source and sink don't lie in the
       * band don't matter */
      if (!pluto_stmt_is_member_of(prog->stmts[dep->src]->id, loop->stmts,
                                   loop->nstmts) ||
          !pluto_stmt_is_member_of(prog->stmts[dep->dest]->id, loop->stmts,
                                   loop->nstmts))
        continue;
      assert(dep->satvec != NULL);
      /* Dependences satisfied outer to the band don't matter */
      if (dep->satisfaction_level < loop->depth)
        continue;
      /* The loop (or scalar dimension) has to be parallel */
      if (dep->satvec[depth])
        break;
    }
    if (i < prog->ndeps)
      break;
    depth++;
  } while (depth < prog->num_hyperplanes);

  /* Peel off scalar dimensions from the end */
  while (depth >= loop->depth + 1 && pluto_is_depth_scalar(loop, depth - 1))
    depth--;

  d = loop->depth;
  while (d <= depth - 2 && !pluto_is_depth_scalar(loop, d + 1))
    d++;
  *innermost_split_level = d;

  width = depth - loop->depth;

  return pluto_band_alloc(loop, width);
}

/* Returns subset of these bands that are not dominated by any other band */
Band **pluto_get_dominator_bands(Band **bands, unsigned nbands,
                                 unsigned *ndom_bands) {
  Band **dom_bands;
  unsigned i, j, ndbands;

  dom_bands = NULL;
  ndbands = 0;
  for (i = 0; i < nbands; i++) {
    for (j = 0; j < nbands; j++) {
      if (is_band_dominated(bands[i], bands[j]))
        break;
    }
    if (j == nbands) {
      Band *dup = pluto_band_dup(bands[i]);
      dom_bands = pluto_bands_cat(&dom_bands, ndbands, &dup, 1);
      ndbands++;
    }
  }
  *ndom_bands = ndbands;

  return dom_bands;
}

/*
 * Return dominating parallel bands
 */
Band **pluto_get_dom_parallel_bands(PlutoProg *prog, unsigned *nbands,
                                    int **comm_placement_levels) {
  Ploop **loops;
  unsigned num, i;
  Band **bands;

  num = 0;
  loops = pluto_get_dom_parallel_loops(prog, &num);

  *comm_placement_levels = (int *)malloc(num * sizeof(int));

  bands = NULL;
  int nb = 0;
  for (i = 0; i < num; i++) {
    int comm_placement_level;
    Band *band = pluto_get_parallel_band(loops[i], prog, &comm_placement_level);
    if (band != NULL) {
      bands = (Band **)realloc(bands, (nb + 1) * sizeof(Band *));
      (*comm_placement_levels)[nb] = comm_placement_level;
      bands[nb++] = band;
    }
  }
  *nbands = nb;

  printf("parallel dominating bands:\n");
  pluto_bands_print(bands, *nbands);
  for (i = 0; i < nb; i++) {
    printf("comm placement level(s): %d\n", (*comm_placement_levels)[i]);
  }

  return bands;
}

/* Set of outermost non-trivial permutable bands (of width >= 2) */
Band **pluto_get_outermost_permutable_bands(PlutoProg *prog,
                                            unsigned *ndbands) {
  Ploop **loops;
  unsigned num, i, nbands;
  Band **bands, **dbands;

  num = 0;
  loops = pluto_get_all_loops(prog, &num);

  bands = NULL;
  nbands = 0;
  for (i = 0; i < num; i++) {
    Band *band = pluto_get_permutable_band(loops[i], prog);
    if (band != NULL) {
      bands = (Band **)realloc(bands, (nbands + 1) * sizeof(Band *));
      bands[nbands++] = band;
    }
  }

  dbands = pluto_get_dominator_bands(bands, nbands, ndbands);

  pluto_loops_free(loops, num);

  pluto_bands_free(bands, nbands);

  return dbands;
}

int pluto_loop_is_innermost(const Ploop *loop, const PlutoProg *prog) {
  unsigned num;
  Ploop **loops;

  loops = pluto_get_loops_under(loop->stmts, loop->nstmts, loop->depth + 1,
                                prog, &num);
  pluto_loops_free(loops, num);

  return (num == 0);
}

/// Does this band have any loops under it. Num levels introduced is the number
/// of scalar dimensions introduced by post tile distribution. The routine
/// returns 1 if the band is innermost. Else it returns false.
int pluto_is_band_innermost(const Band *band, int num_tiling_levels,
                            unsigned num_levels_introduced) {
  for (int i = 0; i < band->loop->nstmts; i++) {
    int firstd = band->loop->depth + (num_tiling_levels + 1) * band->width +
                 num_levels_introduced;
    for (int j = firstd; j < band->loop->stmts[i]->trans->nrows; j++) {
      if (pluto_is_hyperplane_loop(band->loop->stmts[i], j))
        return 0;
    }
  }

  return 1;
}

Ploop **pluto_get_innermost_loops(PlutoProg *prog, int *nloops) {
  Ploop **loops, **iloops;
  unsigned i, num;

  loops = pluto_get_all_loops(prog, &num);

  *nloops = 0;
  iloops = NULL;

  for (i = 0; i < num; i++) {
    if (pluto_loop_is_innermost(loops[i], prog)) {
      Ploop *loop = pluto_loop_dup(loops[i]);
      iloops = pluto_loops_cat(iloops, (*nloops)++, &loop, 1);
    }
  }

  return iloops;
}

/* Set of innermost non-trivial permutable bands (of width >= 2) */
Band **pluto_get_innermost_permutable_bands(const PlutoProg *prog,
                                            unsigned num_tiled_levels,
                                            unsigned *ndbands) {
  Ploop **loops;
  unsigned num, i, nbands;
  Band **bands, **dbands;

  bands = NULL;
  num = 0;
  loops = pluto_get_all_loops(prog, &num);

  nbands = 0;
  for (i = 0; i < num; i++) {
    Band *band = pluto_get_permutable_band(loops[i], prog);
    if (band != NULL && pluto_is_band_innermost(band, num_tiled_levels, 0)) {
      bands = (Band **)realloc(bands, (nbands + 1) * sizeof(Band *));
      bands[nbands] = band;
      nbands++;
    } else
      pluto_band_free(band);
  }
  pluto_loops_free(loops, num);

  dbands = pluto_get_dominator_bands(bands, nbands, ndbands);

  pluto_bands_free(bands, nbands);

  return dbands;
}
