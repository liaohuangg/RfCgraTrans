
   /*+------- <| --------------------------------------------------------**
    **         A                     Clan                                **
    **---     /.\   -----------------------------------------------------**
    **   <|  [""M#                 macros.h                              **
    **-   A   | #   -----------------------------------------------------**
    **   /.\ [""M#         First version: 30/04/2008                     **
    **- [""M# | #  U"U#U  -----------------------------------------------**
         | #  | #  \ .:/
         | #  | #___| #
 ******  | "--'     .-"  ******************************************************
 *     |"-"-"-"-"-#-#-##   Clan : the Chunky Loop Analyzer (experimental)     *
 ****  |     # ## ######  *****************************************************
 *      \       .::::'/                                                       *
 *       \      ::::'/     Copyright (C) 2008 University Paris-Sud 11         *
 *     :8a|    # # ##                                                         *
 *     ::88a      ###      This is free software; you can redistribute it     *
 *    ::::888a  8a ##::.   and/or modify it under the terms of the GNU Lesser *
 *  ::::::::888a88a[]:::   General Public License as published by the Free    *
 *::8:::::::::SUNDOGa8a::. Software Foundation, either version 2.1 of the     *
 *::::::::8::::888:Y8888:: License, or (at your option) any later version.    *
 *::::':::88::::888::Y88a::::::::::::...                                      *
 *::'::..    .   .....   ..   ...  .                                          *
 * This software is distributed in the hope that it will be useful, but       *
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY *
 * or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License   *
 * for more details.							      *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with software; if not, write to the Free Software Foundation, Inc.,  *
 * 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA                     *
 *                                                                            *
 * Clan, the Chunky Loop Analyzer                                             *
 * Written by Cedric Bastoul, Cedric.Bastoul@u-psud.fr                        *
 *                                                                            *
 ******************************************************************************/


/*+****************************************************************************
 *  THIS FILE HAS BEEN AUTOMATICALLY GENERATED FROM macros.h.in BY configure  *
 ******************************************************************************/

#ifndef CLAN_MACROS_H
# define CLAN_MACROS_H

# define CLAN_VERSION                   "0.8.1"

# define CLAN_DEBUG			0
# define CLAN_TRUE			1
# define CLAN_FALSE			0
# define CLAN_MAX_LOCAL_VARIABLES     100
# define CLAN_MAX_STRING             2048
# define CLAN_MAX_DEPTH                50 // Max loop + if depth
# define CLAN_MAX_PARAMETERS          128 // Max parameter number
# define CLAN_MAX_LOCAL_DIMS          128 // Max local dims number
# define CLAN_MAX_CONSTRAINTS         256 // Max contraint number for a domain
# define CLAN_MAX_SCOPS               256 // Max number of SCoPs for autoscop
# define CLAN_MAX_XFOR_INDICES         10 // Max number of indices in (x)for
# define CLAN_UNDEFINED                -1
# define CLAN_KEY_START                 1

# define CLAN_TYPE_ITERATOR             1
# define CLAN_TYPE_ITERATOR_DEC         2
# define CLAN_TYPE_PARAMETER            3
# define CLAN_TYPE_ARRAY                4
# define CLAN_TYPE_LOCAL_DIMS           5
# define CLAN_TYPE_STRUCTURE            6
# define CLAN_TYPE_FIELD                7
# define CLAN_TYPE_FUNCTION             8
# define CLAN_TYPE_READ                 9
# define CLAN_TYPE_WRITE               10
# define CLAN_TYPE_RDWR                11

/* This is a bit of a hack! See clan_options_autopragma_file()! */
# define CLAN_AUTOPRAGMA_FILE clan_options_autopragma_file()

/*---------------------------------------------------------------------------+
 |                               UTILITY MACROS                              |
 +---------------------------------------------------------------------------*/

# define CLAN_info(msg)                                                    \
         do {                                                              \
           fprintf(stderr,"[Clan] Info: " msg " (%s).\n", __func__);       \
         } while (0)

# define CLAN_debug(msg)                                                   \
         do {                                                              \
           if (CLAN_DEBUG)                                                 \
             fprintf(stderr,"[Clan] Debug: " msg " (%s).\n", __func__);    \
         } while (0)

# define CLAN_debug_call(function_call)                                    \
         do {                                                              \
           if (CLAN_DEBUG)                                                 \
             function_call;                                                \
         } while (0)

# define CLAN_warning(msg)                                                 \
         do {                                                              \
           fprintf(stderr,"[Clan] Warning: " msg " (%s).\n", __func__);    \
         } while (0)

# define CLAN_error(msg)                                                   \
         do {                                                              \
           fprintf(stderr,"[Clan] Error: " msg " (%s).\n", __func__);      \
           exit(1);                                                        \
         } while (0)

# define CLAN_malloc(ptr, type, size)                                      \
         do {                                                              \
           if (((ptr) = (type)malloc(size)) == NULL)                       \
             CLAN_error("memory overflow");                                \
         } while (0)

# define CLAN_realloc(ptr, type, size)                                     \
         do {                                                              \
           if (((ptr) = (type)realloc(ptr, size)) == NULL)                 \
             CLAN_error("memory overflow");                                \
         } while (0)

# define CLAN_strdup(destination, source)                                  \
         do {                                                              \
           if (source != NULL) {                                           \
             if (((destination) = strdup(source)) == NULL)                 \
               CLAN_error("memory overflow");                              \
           }                                                               \
           else {                                                          \
             destination = NULL;                                           \
             CLAN_debug("strdup of a NULL string");                        \
           }                                                               \
         } while (0)

# define CLAN_max(x,y) ((x) > (y)? (x) : (y))

# define CLAN_min(x,y) ((x) < (y)? (x) : (y))

#endif /* define CLAN_MACROS_H */
