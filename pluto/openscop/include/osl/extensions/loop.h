/******************************************************************************
 **                            OpenScop Library                              **
 **--------------------------------------------------------------------------**
 **                            extensions/loop.h                             **
 **--------------------------------------------------------------------------**
 **                        First version: 03/06/2013                         **
 ******************************************************************************/

/******************************************************************************
 * OpenScop: Structures and formats for polyhedral tools to talk together     *
 ******************************************************************************
 *    ,___,,_,__,,__,,__,,__,,_,__,,_,__,,__,,___,_,__,,_,__,                 *
 *    /   / /  //  //  //  // /   / /  //  //   / /  // /  /|,_,              *
 *   /   / /  //  //  //  // /   / /  //  //   / /  // /  / / /\              *
 *  |~~~|~|~~~|~~~|~~~|~~~|~|~~~|~|~~~|~~~|~~~|~|~~~|~|~~~|/_/  \             *
 *  | G |C| P | = | L | P |=| = |C| = | = | = |=| = |=| C |\  \ /\            *
 *  | R |l| o | = | e | l |=| = |a| = | = | = |=| = |=| L | \# \ /\           *
 *  | A |a| l | = | t | u |=| = |n| = | = | = |=| = |=| o | |\# \  \          *
 *  | P |n| l | = | s | t |=| = |d| = | = | = | |   |=| o | | \# \  \         *
 *  | H | | y |   | e | o | | = |l|   |   | = | |   | | G | |  \  \  \        *
 *  | I | |   |   | e |   | |   | |   |   |   | |   | |   | |   \  \  \       *
 *  | T | |   |   |   |   | |   | |   |   |   | |   | |   | |    \  \  \      *
 *  | E | |   |   |   |   | |   | |   |   |   | |   | |   | |     \  \  \     *
 *  | * |*| * | * | * | * |*| * |*| * | * | * |*| * |*| * | /      \* \  \    *
 *  | O |p| e | n | S | c |o| p |-| L | i | b |r| a |r| y |/        \  \ /    *
 *  '---'-'---'---'---'---'-'---'-'---'---'---'-'---'-'---'          '--'     *
 *                                                                            *
 * Copyright (C) 2008 University Paris-Sud 11 and INRIA                       *
 *                                                                            *
 * (3-clause BSD license)                                                     *
 * Redistribution and use in source  and binary forms, with or without        *
 * modification, are permitted provided that the following conditions         *
 * are met:                                                                   *
 *                                                                            *
 * 1. Redistributions of source code must retain the above copyright notice,  *
 *    this list of conditions and the following disclaimer.                   *
 * 2. Redistributions in binary form must reproduce the above copyright       *
 *    notice, this list of conditions and the following disclaimer in the     *
 *    documentation and/or other materials provided with the distribution.    *
 * 3. The name of the author may not be used to endorse or promote products   *
 *    derived from this software without specific prior written permission.   *
 *                                                                            *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR       *
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES  *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.    *
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,           *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT   *
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,  *
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY      *
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT        *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF   *
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.          *
 *                                                                            *
 * OpenScop Library, a library to manipulate OpenScop formats and data        *
 * structures. Written by:                                                    *
 * Cedric Bastoul     <Cedric.Bastoul@u-psud.fr> and                          *
 * Louis-Noel Pouchet <Louis-Noel.pouchet@inria.fr>                           *
 *                                                                            *
 ******************************************************************************/

#ifndef OSL_LOOP_H
#define OSL_LOOP_H

#include <stdio.h>

#include <osl/attributes.h>
#include <osl/interface.h>
#include <osl/strings.h>

#if defined(__cplusplus)
extern "C" {
#endif

#define OSL_URI_LOOP "loop"

// loop direcrives
#define OSL_LOOP_DIRECTIVE_NONE 0
#define OSL_LOOP_DIRECTIVE_PARALLEL 1
#define OSL_LOOP_DIRECTIVE_MPI 2
#define OSL_LOOP_DIRECTIVE_VECTOR 4
#define OSL_LOOP_DIRECTIVE_USER 8

/**
 * The osl_loop_t structure stores information about loops in the program
 * in the extension part of the OpenScop representation. Containing the
 * information about the statements in the loop, its iterator and openmp
 * directives, it serves to communicate such information among different
 * tools in the polyhedral chain.
 */
struct osl_loop {
  char* iter;         /**< \brief \0 terminated iterator name */
  size_t nb_stmts;    /**< \brief Number of statements in the loop */
  int* stmt_ids;      /**< \brief Array of statement identifiers. */
  char* private_vars; /**< \brief \0 terminated variable names */
  int directive;      /**< \brief Loop directive to implement */
  char* user;         /**< \brief \0 terminated user string */
  struct osl_loop* next;
};
typedef struct osl_loop osl_loop_t;
typedef struct osl_loop* osl_loop_p;

/******************************************************************************
 *                          Structure display function                        *
 ******************************************************************************/

void osl_loop_idump(FILE*, const osl_loop_t*, int) OSL_NONNULL_ARGS(1);
void osl_loop_dump(FILE*, const osl_loop_t*) OSL_NONNULL_ARGS(1);
char* osl_loop_sprint(const osl_loop_t*) OSL_WARN_UNUSED_RESULT;

/******************************************************************************
 *                               Reading function                             *
 ******************************************************************************/

osl_loop_t* osl_loop_sread(char**) OSL_WARN_UNUSED_RESULT;

/******************************************************************************
 *                    Memory allocation/deallocation function                 *
 ******************************************************************************/

osl_loop_t* osl_loop_malloc(void) OSL_WARN_UNUSED_RESULT;
void osl_loop_free(osl_loop_t*);

/******************************************************************************
 *                            Processing functions                            *
 ******************************************************************************/

osl_loop_t* osl_loop_clone_one(const osl_loop_t*) OSL_WARN_UNUSED_RESULT;
osl_loop_t* osl_loop_clone(const osl_loop_t*) OSL_WARN_UNUSED_RESULT;
bool osl_loop_equal_one(const osl_loop_t*, const osl_loop_t*);
bool osl_loop_equal(const osl_loop_t*, const osl_loop_t*);
osl_strings_t* osl_loop_to_strings(const osl_loop_t*) OSL_WARN_UNUSED_RESULT;
osl_interface_t* osl_loop_interface(void) OSL_WARN_UNUSED_RESULT;

void osl_loop_add(osl_loop_t*, osl_loop_t**);
int osl_loop_count(const osl_loop_t*);

#if defined(__cplusplus)
}
#endif

#endif /* define OSL_LOOP_H */
