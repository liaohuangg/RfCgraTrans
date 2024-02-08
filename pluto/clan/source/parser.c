/* A Bison parser, made by GNU Bison 3.5.1.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2020 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Undocumented macros, especially those whose name start with YY_,
   are private implementation details.  Do not rely on them.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.5.1"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* First part of user prologue.  */
#line 46 "source/parser.y"

   #include <stdio.h>
   #include <stdlib.h>
   #include <string.h>
   #include <assert.h>
   
   #include <osl/macros.h>
   #include <osl/int.h>
   #include <osl/vector.h>
   #include <osl/relation.h>
   #include <osl/statement.h>
   #include <osl/strings.h>
   #include <osl/generic.h>
   #include <osl/body.h>
   #include <osl/extensions/arrays.h>
   #include <osl/extensions/extbody.h>
   #include <osl/scop.h>
   #include <clan/macros.h>
   #include <clan/vector.h>
   #include <clan/relation.h>
   #include <clan/relation_list.h>
   #include <clan/domain.h>
   #include <clan/scop.h>
   #include <clan/symbol.h>
   #include <clan/statement.h>
   #include <clan/options.h>

   int  yylex(void);
   void yyerror(char*);
   void yyrestart(FILE*);
   void clan_scanner_initialize();
   void clan_scanner_reinitialize(int, int, int);
   void clan_scanner_free();

   void clan_parser_add_ld();
   int  clan_parser_nb_ld();
   void clan_parser_log(char*);
   void clan_parser_increment_loop_depth();
   void clan_parser_state_print(FILE*);
   int  clan_parser_is_loop_sane(osl_relation_list_p,osl_relation_list_p,int*);
   void clan_parser_state_initialize(clan_options_p);
   osl_scop_p clan_parse(FILE*, clan_options_p);

   extern FILE*   yyin;                 /**< File to be read by Lex */
   extern int     scanner_parsing;      /**< Do we parse or not? */
   extern char*   scanner_latest_text;  /**< Latest text read by Lex */
   extern char*   scanner_clay;         /**< Data for the Clay software */
   extern int     scanner_line;         /**< Current scanned line */
   extern int     scanner_column;       /**< Scanned column (current) */
   extern int     scanner_column_LALR;  /**< Scanned column (before token) */
   extern int     scanner_scop_start;   /**< Scanned SCoP starting line */
   extern int     scanner_scop_end;     /**< Scanned SCoP ending line */
   extern int     scanner_pragma;       /**< Between SCoP pragmas or not? */

   // This is the "parser state", a collection of variables that vary
   // during the parsing and thanks to we can extract all SCoP informations.
   osl_scop_p     parser_scop;          /**< SCoP in construction */
   clan_symbol_p  parser_symbol;        /**< Top of the symbol table */
   int            parser_recording;     /**< Boolean: do we record or not? */
   char*          parser_record;        /**< What we record (statement body)*/
   int            parser_loop_depth;    /**< Current loop depth */
   int            parser_if_depth;      /**< Current if depth */
   int*           parser_scattering;    /**< Current statement scattering */
   clan_symbol_p* parser_iterators;     /**< Current iterator list */
   clan_domain_p  parser_stack;         /**< Iteration domain stack */
   int*           parser_nb_local_dims; /**< Nb of local dims per depth */
   int            parser_nb_parameters; /**< Nb of parameter symbols */
   int*           parser_valid_else;    /**< Boolean: OK for else per depth */
   int            parser_indent;        /**< SCoP indentation */
   int            parser_error;         /**< Boolean: parse error */

   int            parser_xfor_nb_nests; /**< Current number of xfor nests */
   int*           parser_xfor_depths;   /**< Current xfor nest depth list */
   int*           parser_xfor_labels;   /**< Current xfor label list */
   int            parser_xfor_index;    /**< Nb of current (x)for loop index */
   int*           parser_ceild;         /**< Booleans: ith index used ceild */
   int*           parser_floord;        /**< Booleans: ith index used floord */
   int*           parser_min;           /**< Booleans: ith index used min */
   int*           parser_max;           /**< Booleans: ith index used max */

   // Autoscop-relative variables.
   int            parser_autoscop;      /**< Boolean: autoscop in progress */
   int            parser_line_start;    /**< Autoscop start line, inclusive */
   int            parser_line_end;      /**< Autoscop end line, inclusive */
   int            parser_column_start;  /**< Autoscop start column, inclus. */
   int            parser_column_end;    /**< Autoscop end column, exclusive */

   // Ugly global variable to keep/read Clan options during parsing.
   clan_options_p parser_options;

   // Variables to generate the extbody
   osl_extbody_p  parser_access_extbody; /**< The extbody struct */
   int            parser_access_start;   /**< Start coordinates */
   int            parser_access_length;  /**< Length of the access string*/

#line 166 "source/parser.c"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Use api.header.include to #include this header
   instead of duplicating it here.  */
#ifndef YY_YY_SOURCE_PARSER_H_INCLUDED
# define YY_YY_SOURCE_PARSER_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    CONSTANT = 258,
    STRING_LITERAL = 259,
    SIZEOF = 260,
    PTR_OP = 261,
    INC_OP = 262,
    DEC_OP = 263,
    LEFT_OP = 264,
    RIGHT_OP = 265,
    LE_OP = 266,
    GE_OP = 267,
    EQ_OP = 268,
    NE_OP = 269,
    AND_OP = 270,
    OR_OP = 271,
    MUL_ASSIGN = 272,
    DIV_ASSIGN = 273,
    MOD_ASSIGN = 274,
    ADD_ASSIGN = 275,
    SUB_ASSIGN = 276,
    LEFT_ASSIGN = 277,
    RIGHT_ASSIGN = 278,
    AND_ASSIGN = 279,
    XOR_ASSIGN = 280,
    OR_ASSIGN = 281,
    TYPE_NAME = 282,
    TYPEDEF = 283,
    EXTERN = 284,
    STATIC = 285,
    AUTO = 286,
    REGISTER = 287,
    INLINE = 288,
    RESTRICT = 289,
    CHAR = 290,
    SHORT = 291,
    INT = 292,
    LONG = 293,
    SIGNED = 294,
    UNSIGNED = 295,
    FLOAT = 296,
    DOUBLE = 297,
    CONST = 298,
    VOLATILE = 299,
    VOID = 300,
    BOOL = 301,
    COMPLEX = 302,
    IMAGINARY = 303,
    STRUCT = 304,
    UNION = 305,
    ENUM = 306,
    ELLIPSIS = 307,
    CASE = 308,
    DEFAULT = 309,
    IF = 310,
    ELSE = 311,
    SWITCH = 312,
    WHILE = 313,
    DO = 314,
    XFOR = 315,
    FOR = 316,
    GOTO = 317,
    CONTINUE = 318,
    BREAK = 319,
    RETURN = 320,
    IGNORE = 321,
    PRAGMA = 322,
    MIN = 323,
    MAX = 324,
    CEILD = 325,
    FLOORD = 326,
    ID = 327,
    INTEGER = 328
  };
#endif
/* Tokens.  */
#define CONSTANT 258
#define STRING_LITERAL 259
#define SIZEOF 260
#define PTR_OP 261
#define INC_OP 262
#define DEC_OP 263
#define LEFT_OP 264
#define RIGHT_OP 265
#define LE_OP 266
#define GE_OP 267
#define EQ_OP 268
#define NE_OP 269
#define AND_OP 270
#define OR_OP 271
#define MUL_ASSIGN 272
#define DIV_ASSIGN 273
#define MOD_ASSIGN 274
#define ADD_ASSIGN 275
#define SUB_ASSIGN 276
#define LEFT_ASSIGN 277
#define RIGHT_ASSIGN 278
#define AND_ASSIGN 279
#define XOR_ASSIGN 280
#define OR_ASSIGN 281
#define TYPE_NAME 282
#define TYPEDEF 283
#define EXTERN 284
#define STATIC 285
#define AUTO 286
#define REGISTER 287
#define INLINE 288
#define RESTRICT 289
#define CHAR 290
#define SHORT 291
#define INT 292
#define LONG 293
#define SIGNED 294
#define UNSIGNED 295
#define FLOAT 296
#define DOUBLE 297
#define CONST 298
#define VOLATILE 299
#define VOID 300
#define BOOL 301
#define COMPLEX 302
#define IMAGINARY 303
#define STRUCT 304
#define UNION 305
#define ENUM 306
#define ELLIPSIS 307
#define CASE 308
#define DEFAULT 309
#define IF 310
#define ELSE 311
#define SWITCH 312
#define WHILE 313
#define DO 314
#define XFOR 315
#define FOR 316
#define GOTO 317
#define CONTINUE 318
#define BREAK 319
#define RETURN 320
#define IGNORE 321
#define PRAGMA 322
#define MIN 323
#define MAX 324
#define CEILD 325
#define FLOORD 326
#define ID 327
#define INTEGER 328

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 145 "source/parser.y"
 int value;                      /**< An integer value */
         int* vecint;                    /**< A vector of integer values */
         char* symbol;                   /**< A string for identifiers */
         osl_vector_p affex;             /**< An affine expression */
         osl_relation_p setex;           /**< A set of affine expressions */
         osl_relation_list_p list;       /**< List of array accesses */
         osl_statement_p stmt;           /**< List of statements */
       

#line 374 "source/parser.c"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (void);

#endif /* !YY_YY_SOURCE_PARSER_H_INCLUDED  */



#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))

/* Stored state numbers (used for stacks). */
typedef yytype_int16 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && ! defined __ICC && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                            \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  7
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   886

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  98
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  98
/* YYNRULES -- Number of rules.  */
#define YYNRULES  268
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  453

#define YYUNDEFTOK  2
#define YYMAXUTOK   328


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    86,     2,     2,     2,    87,    93,     2,
      77,    78,    88,    82,    79,    83,    92,    89,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    74,    80,
      84,    81,    85,    97,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    90,     2,    91,    95,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    75,    96,    76,    94,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,   241,   241,   242,   243,   244,   250,   306,   307,   315,
     315,   329,   330,   331,   332,   333,   333,   362,   361,   406,
     407,   418,   417,   431,   440,   449,   439,   481,   480,   527,
     526,   567,   566,   605,   614,   626,   625,   643,   644,   649,
     656,   666,   677,   686,   701,   702,   703,   704,   705,   707,
     709,   710,   714,   715,   720,   721,   726,   747,   754,   767,
     768,   780,   790,   810,   820,   843,   854,   865,   876,   887,
     910,   919,   934,   951,   957,   970,   976,   988,  1018,  1024,
    1035,  1041,  1047,  1058,  1065,  1089,  1113,  1120,  1129,  1142,
    1148,  1160,  1161,  1166,  1172,  1184,  1190,  1201,  1202,  1203,
    1204,  1205,  1214,  1249,  1251,  1253,  1255,  1261,  1263,  1276,
    1290,  1289,  1304,  1321,  1338,  1365,  1367,  1375,  1377,  1401,
    1403,  1405,  1410,  1411,  1412,  1413,  1414,  1415,  1419,  1420,
    1424,  1426,  1431,  1433,  1438,  1443,  1451,  1453,  1458,  1466,
    1468,  1473,  1481,  1483,  1488,  1493,  1498,  1506,  1508,  1513,
    1521,  1523,  1531,  1533,  1541,  1543,  1551,  1553,  1561,  1563,
    1571,  1573,  1582,  1589,  1619,  1621,  1626,  1627,  1628,  1629,
    1630,  1631,  1632,  1633,  1634,  1635,  1639,  1641,  1649,  1656,
    1656,  1731,  1735,  1736,  1737,  1738,  1739,  1740,  1744,  1745,
    1746,  1747,  1748,  1752,  1753,  1754,  1755,  1756,  1757,  1758,
    1759,  1760,  1761,  1762,  1763,  1767,  1768,  1769,  1773,  1774,
    1778,  1779,  1783,  1787,  1788,  1789,  1790,  1794,  1795,  1799,
    1800,  1801,  1805,  1806,  1807,  1811,  1812,  1816,  1817,  1821,
    1822,  1826,  1827,  1831,  1832,  1833,  1834,  1835,  1836,  1837,
    1841,  1842,  1843,  1844,  1848,  1849,  1854,  1855,  1859,  1860,
    1864,  1865,  1866,  1870,  1871,  1875,  1876,  1880,  1881,  1882,
    1886,  1887,  1888,  1889,  1890,  1891,  1892,  1893,  1894
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "CONSTANT", "STRING_LITERAL", "SIZEOF",
  "PTR_OP", "INC_OP", "DEC_OP", "LEFT_OP", "RIGHT_OP", "LE_OP", "GE_OP",
  "EQ_OP", "NE_OP", "AND_OP", "OR_OP", "MUL_ASSIGN", "DIV_ASSIGN",
  "MOD_ASSIGN", "ADD_ASSIGN", "SUB_ASSIGN", "LEFT_ASSIGN", "RIGHT_ASSIGN",
  "AND_ASSIGN", "XOR_ASSIGN", "OR_ASSIGN", "TYPE_NAME", "TYPEDEF",
  "EXTERN", "STATIC", "AUTO", "REGISTER", "INLINE", "RESTRICT", "CHAR",
  "SHORT", "INT", "LONG", "SIGNED", "UNSIGNED", "FLOAT", "DOUBLE", "CONST",
  "VOLATILE", "VOID", "BOOL", "COMPLEX", "IMAGINARY", "STRUCT", "UNION",
  "ENUM", "ELLIPSIS", "CASE", "DEFAULT", "IF", "ELSE", "SWITCH", "WHILE",
  "DO", "XFOR", "FOR", "GOTO", "CONTINUE", "BREAK", "RETURN", "IGNORE",
  "PRAGMA", "MIN", "MAX", "CEILD", "FLOORD", "ID", "INTEGER", "':'", "'{'",
  "'}'", "'('", "')'", "','", "';'", "'='", "'+'", "'-'", "'<'", "'>'",
  "'!'", "'%'", "'*'", "'/'", "'['", "']'", "'.'", "'&'", "'~'", "'^'",
  "'|'", "'?'", "$accept", "scop_list", "scop", "statement_list",
  "statement_indented", "$@1", "statement", "$@2", "labeled_statement",
  "$@3", "compound_statement", "selection_else_statement", "$@4",
  "selection_statement", "$@5", "$@6", "iteration_statement", "$@7", "$@8",
  "$@9", "loop_initialization_list", "loop_initialization", "$@10",
  "loop_declaration", "loop_condition_list", "loop_condition",
  "loop_stride_list", "loop_stride", "idparent", "loop_infinite",
  "loop_body", "affine_minmax_expression", "minmax",
  "affine_min_expression", "affine_max_expression", "affine_relation",
  "affine_logical_and_expression", "affine_condition",
  "affine_primary_expression", "affine_unary_expression",
  "affine_multiplicative_expression", "affine_expression",
  "affine_ceildfloord_expression", "ceildfloord",
  "affine_ceild_expression", "affine_floord_expression",
  "id_or_clan_keyword", "primary_expression", "postfix_expression", "$@11",
  "argument_expression_list", "unary_expression", "unary_operator",
  "unary_increment_operator", "cast_expression",
  "multiplicative_expression", "additive_expression", "shift_expression",
  "relational_expression", "equality_expression", "and_expression",
  "exclusive_or_expression", "inclusive_or_expression",
  "logical_and_expression", "logical_or_expression",
  "conditional_expression", "assignment_expression", "assignment_operator",
  "assignment_rdwr_operator", "expression", "expression_statement", "$@12",
  "constant_expression", "declaration_specifiers",
  "storage_class_specifier", "type_specifier", "struct_or_union_specifier",
  "struct_or_union", "struct_declaration_list", "struct_declaration",
  "specifier_qualifier_list", "struct_declarator_list",
  "struct_declarator", "enum_specifier", "enumerator_list", "enumerator",
  "type_qualifier", "declarator", "direct_declarator", "pointer",
  "type_qualifier_list", "parameter_type_list", "parameter_list",
  "parameter_declaration", "identifier_list", "type_name",
  "abstract_declarator", "direct_abstract_declarator", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_int16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,    58,   123,   125,    40,    41,    44,
      59,    61,    43,    45,    60,    62,    33,    37,    42,    47,
      91,    93,    46,    38,   126,    94,   124,    63
};
# endif

#define YYPACT_NINF (-364)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-96)

#define yytable_value_is_error(Yyn) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     -16,  -364,    38,  -364,   -14,  -364,   197,  -364,  -364,  -364,
    -364,  -364,   -19,    21,    90,  -364,  -364,   279,  -364,  -364,
    -364,  -364,   414,   792,  -364,  -364,   101,   110,   131,   144,
    -364,  -364,  -364,  -364,   441,  -364,  -364,  -364,  -364,  -364,
    -364,  -364,  -364,   224,  -364,  -364,  -364,  -364,  -364,  -364,
    -364,  -364,    47,   546,   414,   468,  -364,   255,   133,    58,
      70,   267,   130,   104,   141,   235,    30,  -364,  -364,   229,
     170,   179,   194,   199,  -364,  -364,   792,   140,   140,   207,
       1,    15,  -364,   271,    54,  -364,  -364,   269,   137,  -364,
    -364,   197,  -364,   225,   263,    -1,   197,   224,  -364,  -364,
    -364,  -364,  -364,  -364,  -364,  -364,  -364,  -364,  -364,  -364,
    -364,  -364,  -364,   561,   301,   808,  -364,   811,   -26,  -364,
     808,   226,   284,   238,   206,   284,  -364,  -364,  -364,  -364,
    -364,  -364,  -364,  -364,  -364,  -364,  -364,  -364,   414,  -364,
    -364,  -364,   414,  -364,   414,   414,   414,   414,   414,   414,
     414,   414,   414,   414,   414,   414,   414,   414,   414,   414,
     414,   414,   414,   414,  -364,   171,   530,   206,   206,    73,
     122,   206,  -364,  -364,   792,   530,   530,   171,   171,   792,
     792,  -364,   206,   206,   206,   206,   206,   240,  -364,   244,
    -364,   792,   303,   264,   265,   792,  -364,  -364,   291,   284,
     296,  -364,  -364,   808,   317,   611,    -4,   316,   -29,  -364,
      -3,  -364,   414,  -364,  -364,   414,   103,  -364,  -364,  -364,
    -364,  -364,   255,   255,   133,   133,    58,    58,    58,    58,
      70,    70,   267,   130,   104,   141,   235,   -32,  -364,   297,
     308,   324,   308,   208,   220,  -364,  -364,   163,   128,  -364,
    -364,  -364,  -364,  -364,   271,   197,  -364,  -364,   308,   269,
     269,   395,  -364,    64,   315,   407,   263,  -364,  -364,   352,
      64,  -364,   354,    80,  -364,   284,   707,  -364,   261,   808,
    -364,  -364,  -364,  -364,  -364,  -364,    92,   789,   789,   789,
     379,   383,  -364,   403,  -364,  -364,    -4,  -364,  -364,   399,
      -3,   688,   343,  -364,   318,  -364,  -364,   414,   171,   530,
     419,   420,  -364,  -364,   421,    11,    11,  -364,    11,   417,
     422,    24,   792,  -364,  -364,   418,  -364,   425,   414,  -364,
     284,    89,  -364,  -364,   414,   293,  -364,   321,  -364,   424,
     106,   623,   735,   547,  -364,   121,  -364,  -364,  -364,  -364,
    -364,   763,  -364,  -364,  -364,  -364,  -364,   426,  -364,   415,
    -364,   414,  -364,   427,   437,   438,   439,   463,  -364,  -364,
    -364,   442,  -364,    64,  -364,  -364,   448,   449,    11,  -364,
     397,  -364,  -364,  -364,  -364,  -364,   447,   261,  -364,   414,
     636,   370,   106,  -364,  -364,  -364,  -364,  -364,  -364,  -364,
    -364,  -364,  -364,  -364,  -364,  -364,   197,  -364,  -364,  -364,
     323,  -364,  -364,  -364,  -364,  -364,   451,   308,  -364,   453,
     197,  -364,  -364,  -364,  -364,  -364,   454,   349,  -364,   435,
     197,  -364,   458,   460,   397,   206,  -364,  -364,  -364,   284,
    -364,  -364,  -364,  -364,   464,   232,  -364,   397,   469,   466,
     470,  -364,  -364
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_int16 yydefact[] =
{
       9,     5,     9,     2,     9,     7,   179,     1,     4,     3,
       6,     8,     0,     0,     9,   178,    10,     0,    11,    12,
      14,    13,     0,     0,    17,    19,     9,     0,     0,     0,
      16,    31,   103,   105,     0,   128,   129,    98,    99,   100,
     101,    97,   104,     0,   124,   125,   127,   123,   122,   126,
     102,   107,   117,   130,     0,     0,   132,   136,   139,   142,
     147,   150,   152,   154,   156,   158,   160,   162,   176,     0,
       0,     0,     0,     0,    77,    78,     0,     0,     0,     0,
       0,     0,    73,    75,     0,    80,    83,    86,    93,    63,
      61,   179,    20,     0,    38,    38,   179,     0,   120,   204,
     194,   195,   196,   197,   200,   201,   198,   199,   229,   230,
     193,   208,   209,     0,     0,   214,   202,     0,   255,   203,
     216,     0,     0,   110,     0,     0,   114,   166,   167,   168,
     169,   170,   171,   172,   173,   174,   175,   164,     0,   165,
     130,   119,     0,   118,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   180,     0,     0,     0,     0,     0,
      93,     0,    81,    82,     0,     0,     0,     0,     0,     0,
       0,    24,     0,     0,     0,     0,     0,     0,    18,     0,
      37,     0,     0,     0,     0,     0,    56,    32,     0,     0,
     224,   106,   213,     0,   207,     0,   240,     0,   257,   256,
     258,   215,     0,   113,   109,     0,     0,   112,   163,   135,
     133,   134,   137,   138,   140,   141,   145,   146,   143,   144,
     148,   149,   151,   153,   155,   157,   159,     0,   177,     0,
      95,     0,    93,     0,     0,    70,    79,     0,     0,    68,
      66,    67,    65,    74,    76,   179,    84,    85,    69,    87,
      88,     0,    54,     0,     0,    41,    38,    34,    35,     0,
       0,   121,   227,     0,   225,     0,     0,   210,     0,     0,
     188,   189,   190,   191,   192,   265,   252,   182,   184,   186,
       0,   246,   248,     0,   244,   242,   241,   261,   181,     0,
     259,     0,     0,   131,     0,   115,   108,     0,     0,     0,
       0,     0,    71,    25,     0,     0,     0,    52,     0,     0,
      43,     0,     0,    40,    33,     0,    55,     0,     0,   222,
       0,     0,   206,   211,     0,     0,   233,     0,   217,   219,
     232,     0,     0,     0,   250,   257,   251,   183,   185,   187,
     266,     0,   260,   245,   243,   262,   267,     0,   263,     0,
     111,     0,   161,     0,     0,     0,     0,    23,    72,    46,
      47,     0,    27,     0,    44,    45,     0,     0,     0,    39,
       0,    29,   228,   226,   223,   220,     0,     0,   212,     0,
       0,     0,   231,   205,   247,   249,   268,   264,   116,    62,
      64,    94,    96,    21,    26,    53,   179,    42,    50,    51,
       0,    59,    60,    91,    92,    36,     0,    89,    57,     0,
     179,   234,   218,   221,   239,   253,     0,     0,   236,     0,
     179,    28,     0,     0,     0,     0,    30,   237,   238,     0,
     235,    22,    48,    49,     0,     0,   254,     0,     0,     0,
       0,    58,    90
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -364,  -364,   544,   533,    37,  -364,    -6,  -364,  -364,  -364,
    -364,  -364,  -364,  -364,  -364,  -364,  -364,  -364,  -364,  -364,
     -90,  -364,  -364,  -364,  -177,  -364,  -243,  -364,  -240,  -364,
    -363,  -250,  -364,  -156,  -151,   373,   369,   -17,   355,   268,
     274,   -75,  -364,  -364,  -364,  -364,  -105,  -364,  -364,  -364,
    -364,   -20,  -364,   501,   -38,   307,   328,   174,   333,   398,
     400,   401,   413,   396,  -364,  -188,  -135,  -364,  -364,   -15,
    -364,  -364,  -268,    79,  -364,  -172,  -364,  -364,   280,  -211,
      13,  -364,   193,  -364,   306,   275,  -176,  -133,  -272,  -108,
    -364,  -264,  -364,   242,  -364,   497,  -114,  -197
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     2,     3,     4,     5,     6,   196,    17,    18,    91,
      19,   404,   430,    20,   255,   367,    30,   406,   420,    96,
     191,   192,   325,   193,   263,   264,   319,   320,   321,    31,
     197,   415,   416,    80,    81,    82,    83,   265,    85,    86,
      87,    88,   418,   419,    89,    90,    50,    51,    52,   215,
     304,   140,    54,    55,    56,    57,    58,    59,    60,    61,
      62,    63,    64,    65,    66,    67,    68,   138,   139,   114,
      21,    22,   299,   286,   287,   115,   116,   117,   276,   277,
     278,   337,   338,   119,   273,   274,   120,   339,   340,   341,
     296,   290,   291,   292,   427,   121,   293,   210
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      16,   170,    53,   218,   209,   195,    84,    69,   200,   239,
     208,   300,   204,   175,    98,   241,   141,   213,   270,   298,
     217,   251,   252,    53,   249,   250,   177,   327,   238,   289,
     294,   374,   375,   288,   359,   143,   190,   357,     7,   108,
     109,    11,   307,   431,   376,   377,   161,   163,   205,   216,
       1,   205,    10,   122,    35,    36,   118,   436,    23,   169,
     382,   207,   206,    11,   207,   333,   385,   149,   150,   392,
     180,   315,   316,   392,   301,   369,   370,    53,   371,   194,
     305,   151,   152,   317,   206,   188,   176,   302,   318,   180,
     240,   242,   243,   244,   272,    24,   247,   208,   295,   178,
     242,   242,   240,   240,     8,   378,   219,   220,   221,   258,
     118,   289,   289,   289,   298,   288,   288,   288,    53,   362,
     353,   423,    53,   429,   123,   289,   426,   162,   202,   288,
     407,   333,   181,   211,   -95,   184,   317,   124,   410,   125,
     298,   318,    53,    53,   180,   379,   298,   237,   300,   -95,
     184,   245,   363,   344,   153,   154,   329,   248,   364,   330,
      37,    38,    39,    40,    41,   384,    25,   289,   330,   343,
     272,   288,   346,   336,   303,   289,   324,    92,   345,   288,
     206,   336,   207,   390,   444,   185,   186,    93,   354,    37,
      38,    39,    40,    41,   306,    53,   391,   449,   343,   158,
     246,   298,   386,   298,   185,   186,   312,   -95,    94,   187,
     386,   207,    74,    75,   289,   147,   148,   171,   288,   185,
     186,    95,   -95,   157,   187,   272,   398,    32,    33,    34,
     336,    35,    36,   240,   242,   345,   336,   159,   336,    70,
     336,   246,    73,    74,    75,   185,   186,   165,   171,   313,
     160,    99,    12,    77,    78,   -15,   166,   -15,   -15,   100,
     101,   102,   103,   104,   105,   106,   107,   108,   109,   110,
      13,   167,    14,   111,   112,   113,   168,    15,    74,    75,
     155,   156,   336,   171,   174,   425,   179,   310,    77,    78,
     185,   186,    37,    38,    39,    40,    41,    42,   189,   311,
     190,    43,   185,   186,   212,   417,    44,    45,   163,   164,
      46,   448,    47,   261,   185,   186,   214,    48,    49,    32,
      33,    34,   262,    35,    36,   226,   227,   228,   229,    37,
      38,    39,    40,    41,   446,   334,   268,    27,   335,    28,
      29,    53,   144,   145,   146,   269,    32,    33,    34,   206,
      35,    36,    37,    38,    39,    40,    41,   182,   183,   417,
     445,    37,    38,    39,    40,    41,   347,   348,   349,   271,
     335,   275,   417,    32,    33,    34,   308,    35,    36,   201,
     163,   206,   266,   267,    37,    38,    39,    40,    41,    42,
     185,   186,   279,    43,   322,   323,   360,   361,    44,    45,
     387,   388,    46,   309,    47,   432,   433,   297,   314,    48,
      49,    37,    38,    39,    40,    41,    42,    32,    33,    34,
      43,    35,    36,   180,   441,    44,    45,   438,   439,    46,
     326,    47,   172,   173,   358,   328,    48,    49,    37,    38,
      39,    40,    41,    42,    32,    33,    34,    43,    35,    36,
     256,   257,    44,    45,   222,   223,    46,   350,    47,   259,
     260,   428,   351,    48,    49,   411,   412,   413,   414,    74,
      75,    32,    33,    34,   171,    35,    36,   224,   225,    77,
      78,   352,    37,    38,    39,    40,    41,    42,   230,   231,
     355,    43,   365,   366,   368,   372,    44,    45,   389,   380,
      46,   373,    47,   381,   396,   399,   397,    48,    49,    37,
      38,    39,    40,    41,    42,   400,   401,   402,    97,   403,
     405,   408,   409,    44,    45,   421,   440,    46,   434,    47,
     435,   442,   437,   443,    48,    49,    37,    38,    39,    40,
      41,    42,   450,   447,   451,   142,     9,    26,   452,   254,
      44,    45,   253,   126,    46,   232,    47,   236,   233,   342,
     234,    48,    49,   127,   128,   129,   130,   131,   132,   133,
     134,   135,   136,   235,    99,   280,   281,   282,   283,   284,
     422,   331,   100,   101,   102,   103,   104,   105,   106,   107,
     108,   109,   110,   395,   198,     0,   111,   112,   113,    71,
      72,     0,    74,    75,     0,   383,     0,   171,     0,     0,
       0,     0,    77,    78,     0,    37,    38,    39,    40,    41,
       0,     0,     0,     0,   343,   285,     0,   137,     0,    37,
      38,    39,    40,    41,     0,   206,   199,   207,    99,   280,
     281,   282,   283,   284,     0,     0,   100,   101,   102,   103,
     104,   105,   106,   107,   108,   109,   110,     0,     0,     0,
     111,   112,   113,    99,   280,   281,   282,   283,   284,     0,
       0,   100,   101,   102,   103,   104,   105,   106,   107,   108,
     109,   110,     0,     0,     0,   111,   112,   113,   205,   285,
       0,    37,    38,    39,    40,    41,     0,     0,     0,   206,
     335,   207,     0,     0,    37,    38,    39,    40,    41,     0,
       0,     0,     0,     0,   424,    99,   280,   281,   282,   283,
     284,     0,     0,   100,   101,   102,   103,   104,   105,   106,
     107,   108,   109,   110,    99,     0,     0,   111,   112,   113,
       0,     0,   100,   101,   102,   103,   104,   105,   106,   107,
     108,   109,   110,     0,     0,     0,   111,   112,   113,     0,
       0,     0,    99,     0,     0,     0,   356,     0,     0,     0,
     100,   101,   102,   103,   104,   105,   106,   107,   108,   109,
     110,     0,     0,   332,   111,   112,   113,     0,     0,     0,
      99,   280,   281,   282,   283,   284,     0,     0,   100,   101,
     102,   103,   104,   105,   106,   107,   108,   109,   110,     0,
       0,   393,   111,   112,   113,   394,    99,   280,   281,   282,
     283,   284,     0,     0,   100,   101,   102,   103,   104,   105,
     106,   107,   108,   109,   110,    99,     0,     0,   111,   112,
     113,     0,     0,   100,   101,   102,   103,   104,   105,   106,
     107,   108,   109,   110,     0,     0,     0,   111,   112,   113,
      70,    71,    72,    73,    74,    75,     0,     0,     0,    76,
       0,     0,     0,     0,    77,    78,     0,     0,    79,    37,
      38,    39,    40,    41,     0,     0,   203
};

static const yytype_int16 yycheck[] =
{
       6,    76,    22,   138,   118,    95,    23,    22,   113,   165,
     118,   208,   117,    12,    34,   166,    54,   122,   195,   207,
     125,   177,   178,    43,   175,   176,    11,   270,   163,   205,
     206,     7,     8,   205,   302,    55,    37,   301,     0,    43,
      44,     4,    74,   406,    20,    21,    16,    79,    77,   124,
      66,    77,    66,     6,     7,     8,    43,   420,    77,    76,
     328,    90,    88,    26,    90,   276,   334,     9,    10,   341,
      16,     7,     8,   345,    77,   315,   316,    97,   318,    80,
     215,    11,    12,    72,    88,    91,    85,    90,    77,    16,
     165,   166,   167,   168,   199,    74,   171,   205,   206,    84,
     175,   176,   177,   178,    66,    81,   144,   145,   146,   184,
      97,   287,   288,   289,   302,   287,   288,   289,   138,   307,
     296,   389,   142,   391,    77,   301,   390,    97,   115,   301,
     373,   342,    78,   120,    12,    13,    72,    90,   378,    92,
     328,    77,   162,   163,    16,   322,   334,   162,   345,    12,
      13,    78,   308,   286,    84,    85,    76,   174,   309,    79,
      68,    69,    70,    71,    72,    76,    76,   343,    79,    77,
     275,   343,   286,   278,   212,   351,   266,    76,   286,   351,
      88,   286,    90,    77,   434,    82,    83,    77,   296,    68,
      69,    70,    71,    72,    91,   215,    90,   447,    77,    95,
      78,   389,   335,   391,    82,    83,    78,    85,    77,    87,
     343,    90,    72,    73,   390,    82,    83,    77,   390,    82,
      83,    77,    85,    93,    87,   330,   361,     3,     4,     5,
     335,     7,     8,   308,   309,   343,   341,    96,   343,    68,
     345,    78,    71,    72,    73,    82,    83,    77,    77,   255,
      15,    27,    55,    82,    83,    58,    77,    60,    61,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      73,    77,    75,    49,    50,    51,    77,    80,    72,    73,
      13,    14,   387,    77,    77,   390,    15,    79,    82,    83,
      82,    83,    68,    69,    70,    71,    72,    73,    73,    79,
      37,    77,    82,    83,    78,   380,    82,    83,    79,    80,
      86,    79,    88,    73,    82,    83,    78,    93,    94,     3,
       4,     5,    78,     7,     8,   151,   152,   153,   154,    68,
      69,    70,    71,    72,   439,    74,    72,    58,    77,    60,
      61,   361,    87,    88,    89,    80,     3,     4,     5,    88,
       7,     8,    68,    69,    70,    71,    72,    88,    89,   434,
     435,    68,    69,    70,    71,    72,   287,   288,   289,    78,
      77,    75,   447,     3,     4,     5,    79,     7,     8,    78,
      79,    88,    79,    80,    68,    69,    70,    71,    72,    73,
      82,    83,    75,    77,    79,    80,    78,    79,    82,    83,
      79,    80,    86,    79,    88,    82,    83,    91,    13,    93,
      94,    68,    69,    70,    71,    72,    73,     3,     4,     5,
      77,     7,     8,    16,   430,    82,    83,    78,    79,    86,
      78,    88,    77,    78,    91,    81,    93,    94,    68,    69,
      70,    71,    72,    73,     3,     4,     5,    77,     7,     8,
     182,   183,    82,    83,   147,   148,    86,    78,    88,   185,
     186,    91,    79,    93,    94,    68,    69,    70,    71,    72,
      73,     3,     4,     5,    77,     7,     8,   149,   150,    82,
      83,    78,    68,    69,    70,    71,    72,    73,   155,   156,
      91,    77,    73,    73,    73,    78,    82,    83,    74,    81,
      86,    79,    88,    78,    78,    78,    91,    93,    94,    68,
      69,    70,    71,    72,    73,    78,    78,    78,    77,    56,
      78,    73,    73,    82,    83,    78,    91,    86,    77,    88,
      77,    73,    78,    73,    93,    94,    68,    69,    70,    71,
      72,    73,    73,    79,    78,    77,     2,    14,    78,   180,
      82,    83,   179,    52,    86,   157,    88,   161,   158,   279,
     159,    93,    94,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,   160,    27,    28,    29,    30,    31,    32,
     387,   275,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,   351,    97,    -1,    49,    50,    51,    69,
      70,    -1,    72,    73,    -1,   330,    -1,    77,    -1,    -1,
      -1,    -1,    82,    83,    -1,    68,    69,    70,    71,    72,
      -1,    -1,    -1,    -1,    77,    78,    -1,    81,    -1,    68,
      69,    70,    71,    72,    -1,    88,    75,    90,    27,    28,
      29,    30,    31,    32,    -1,    -1,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    -1,    -1,    -1,
      49,    50,    51,    27,    28,    29,    30,    31,    32,    -1,
      -1,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    -1,    -1,    -1,    49,    50,    51,    77,    78,
      -1,    68,    69,    70,    71,    72,    -1,    -1,    -1,    88,
      77,    90,    -1,    -1,    68,    69,    70,    71,    72,    -1,
      -1,    -1,    -1,    -1,    78,    27,    28,    29,    30,    31,
      32,    -1,    -1,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    27,    -1,    -1,    49,    50,    51,
      -1,    -1,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    -1,    -1,    -1,    49,    50,    51,    -1,
      -1,    -1,    27,    -1,    -1,    -1,    78,    -1,    -1,    -1,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    -1,    -1,    76,    49,    50,    51,    -1,    -1,    -1,
      27,    28,    29,    30,    31,    32,    -1,    -1,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    -1,
      -1,    76,    49,    50,    51,    52,    27,    28,    29,    30,
      31,    32,    -1,    -1,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    27,    -1,    -1,    49,    50,
      51,    -1,    -1,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    -1,    -1,    -1,    49,    50,    51,
      68,    69,    70,    71,    72,    73,    -1,    -1,    -1,    77,
      -1,    -1,    -1,    -1,    82,    83,    -1,    -1,    86,    68,
      69,    70,    71,    72,    -1,    -1,    75
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    66,    99,   100,   101,   102,   103,     0,    66,   100,
      66,   102,    55,    73,    75,    80,   104,   105,   106,   108,
     111,   168,   169,    77,    74,    76,   101,    58,    60,    61,
     114,   127,     3,     4,     5,     7,     8,    68,    69,    70,
      71,    72,    73,    77,    82,    83,    86,    88,    93,    94,
     144,   145,   146,   149,   150,   151,   152,   153,   154,   155,
     156,   157,   158,   159,   160,   161,   162,   163,   164,   167,
      68,    69,    70,    71,    72,    73,    77,    82,    83,    86,
     131,   132,   133,   134,   135,   136,   137,   138,   139,   142,
     143,   107,    76,    77,    77,    77,   117,    77,   149,    27,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    49,    50,    51,   167,   173,   174,   175,   178,   181,
     184,   193,     6,    77,    90,    92,   151,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    81,   165,   166,
     149,   152,    77,   149,    87,    88,    89,    82,    83,     9,
      10,    11,    12,    84,    85,    13,    14,    93,    95,    96,
      15,    16,    97,    79,    80,    77,    77,    77,    77,   135,
     139,    77,   136,   136,    77,    12,    85,    11,    84,    15,
      16,    78,    88,    89,    13,    82,    83,    87,   104,    73,
      37,   118,   119,   121,    80,   118,   104,   128,   193,    75,
     144,    78,   178,    75,   144,    77,    88,    90,   187,   194,
     195,   178,    78,   144,    78,   147,   139,   144,   164,   152,
     152,   152,   153,   153,   154,   154,   155,   155,   155,   155,
     156,   156,   157,   158,   159,   160,   161,   167,   164,   131,
     139,   132,   139,   139,   139,    78,    78,   139,   135,   132,
     132,   131,   131,   133,   134,   112,   137,   137,   139,   138,
     138,    73,    78,   122,   123,   135,    79,    80,    72,    80,
     122,    78,   144,   182,   183,    75,   176,   177,   178,    75,
      28,    29,    30,    31,    32,    78,   171,   172,   173,   184,
     189,   190,   191,   194,   184,   187,   188,    91,   163,   170,
     195,    77,    90,   152,   148,   164,    91,    74,    79,    79,
      79,    79,    78,   104,    13,     7,     8,    72,    77,   124,
     125,   126,    79,    80,   118,   120,    78,   124,    81,    76,
      79,   182,    76,   177,    74,    77,   144,   179,   180,   185,
     186,   187,   176,    77,   185,   187,   194,   171,   171,   171,
      78,    79,    78,   184,   187,    91,    78,   189,    91,   170,
      78,    79,   163,   131,   132,    73,    73,   113,    73,   126,
     126,   126,    78,    79,     7,     8,    20,    21,    81,   122,
      81,    78,   170,   183,    76,   170,   185,    79,    80,    74,
      77,    90,   186,    76,    52,   191,    78,    91,   164,    78,
      78,    78,    78,    56,   109,    78,   115,   124,    73,    73,
     126,    68,    69,    70,    71,   129,   130,   139,   140,   141,
     116,    78,   180,   170,    78,   144,   189,   192,    91,   170,
     110,   128,    82,    83,    77,    77,   128,    78,    78,    79,
      91,   104,    73,    73,   129,   139,   144,    79,    79,   129,
      73,    78,    78
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    98,    99,    99,    99,    99,   100,   101,   101,   103,
     102,   104,   104,   104,   104,   105,   104,   107,   106,   108,
     108,   110,   109,   109,   112,   113,   111,   115,   114,   116,
     114,   117,   114,   118,   118,   120,   119,   121,   121,   122,
     122,   123,   124,   124,   125,   125,   125,   125,   125,   125,
     125,   125,   126,   126,   127,   127,   128,   129,   129,   130,
     130,   131,   131,   132,   132,   133,   133,   133,   133,   133,
     133,   133,   133,   134,   134,   135,   135,   136,   136,   136,
     137,   137,   137,   138,   138,   138,   139,   139,   139,   140,
     140,   141,   141,   142,   142,   143,   143,   144,   144,   144,
     144,   144,   145,   145,   145,   145,   145,   146,   146,   146,
     147,   146,   146,   146,   146,   148,   148,   149,   149,   149,
     149,   149,   150,   150,   150,   150,   150,   150,   151,   151,
     152,   152,   153,   153,   153,   153,   154,   154,   154,   155,
     155,   155,   156,   156,   156,   156,   156,   157,   157,   157,
     158,   158,   159,   159,   160,   160,   161,   161,   162,   162,
     163,   163,   164,   164,   165,   165,   166,   166,   166,   166,
     166,   166,   166,   166,   166,   166,   167,   167,   168,   169,
     168,   170,   171,   171,   171,   171,   171,   171,   172,   172,
     172,   172,   172,   173,   173,   173,   173,   173,   173,   173,
     173,   173,   173,   173,   173,   174,   174,   174,   175,   175,
     176,   176,   177,   178,   178,   178,   178,   179,   179,   180,
     180,   180,   181,   181,   181,   182,   182,   183,   183,   184,
     184,   185,   185,   186,   186,   186,   186,   186,   186,   186,
     187,   187,   187,   187,   188,   188,   189,   189,   190,   190,
     191,   191,   191,   192,   192,   193,   193,   194,   194,   194,
     195,   195,   195,   195,   195,   195,   195,   195,   195
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     1,     2,     2,     1,     2,     1,     2,     0,
       2,     1,     1,     1,     1,     0,     2,     0,     4,     2,
       3,     0,     3,     0,     0,     0,     8,     0,     8,     0,
       8,     0,     3,     3,     2,     0,     5,     1,     0,     3,
       2,     1,     3,     1,     2,     2,     2,     2,     5,     5,
       3,     3,     1,     3,     4,     5,     1,     1,     6,     1,
       1,     1,     6,     1,     6,     3,     3,     3,     3,     3,
       3,     4,     5,     1,     3,     1,     3,     1,     1,     3,
       1,     2,     2,     1,     3,     3,     1,     3,     3,     1,
       6,     1,     1,     1,     6,     1,     6,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     3,     1,     4,     3,
       0,     5,     3,     3,     2,     1,     3,     1,     2,     2,
       2,     4,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     4,     1,     3,     3,     3,     1,     3,     3,     1,
       3,     3,     1,     3,     3,     3,     3,     1,     3,     3,
       1,     3,     1,     3,     1,     3,     1,     3,     1,     3,
       1,     5,     1,     3,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     3,     1,     0,
       3,     1,     1,     2,     1,     2,     1,     2,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     5,     4,     2,     1,     1,
       1,     2,     3,     2,     1,     2,     1,     1,     3,     1,
       2,     3,     4,     5,     2,     1,     3,     1,     3,     1,
       1,     2,     1,     1,     3,     4,     3,     4,     4,     3,
       1,     2,     2,     3,     1,     2,     1,     3,     1,     3,
       2,     2,     1,     1,     3,     1,     2,     1,     1,     2,
       3,     2,     3,     3,     4,     2,     3,     3,     4
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256



/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)

/* This macro is provided for backward compatibility. */
#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo, int yytype, YYSTYPE const * const yyvaluep)
{
  FILE *yyoutput = yyo;
  YYUSE (yyoutput);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyo, yytoknum[yytype], *yyvaluep);
# endif
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo, int yytype, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyo, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  yy_symbol_value_print (yyo, yytype, yyvaluep);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp, int yyrule)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[+yyssp[yyi + 1 - yynrhs]],
                       &yyvsp[(yyi + 1) - (yynrhs)]
                                              );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen(S) (YY_CAST (YYPTRDIFF_T, strlen (S)))
#  else
/* Return the length of YYSTR.  */
static YYPTRDIFF_T
yystrlen (const char *yystr)
{
  YYPTRDIFF_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYPTRDIFF_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYPTRDIFF_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            else
              goto append;

          append:
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (yyres)
    return yystpcpy (yyres, yystr) - yyres;
  else
    return yystrlen (yystr);
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYPTRDIFF_T *yymsg_alloc, char **yymsg,
                yy_state_t *yyssp, int yytoken)
{
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat: reported tokens (one for the "unexpected",
     one per "expected"). */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Actual size of YYARG. */
  int yycount = 0;
  /* Cumulated lengths of YYARG.  */
  YYPTRDIFF_T yysize = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[+*yyssp];
      YYPTRDIFF_T yysize0 = yytnamerr (YY_NULLPTR, yytname[yytoken]);
      yysize = yysize0;
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                {
                  YYPTRDIFF_T yysize1
                    = yysize + yytnamerr (YY_NULLPTR, yytname[yyx]);
                  if (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM)
                    yysize = yysize1;
                  else
                    return 2;
                }
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
    default: /* Avoid compiler warnings. */
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  {
    /* Don't count the "%s"s in the final size, but reserve room for
       the terminator.  */
    YYPTRDIFF_T yysize1 = yysize + (yystrlen (yyformat) - 2 * yycount) + 1;
    if (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM)
      yysize = yysize1;
    else
      return 2;
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          ++yyp;
          ++yyformat;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
{
  YYUSE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  switch (yytype)
    {
    case 72: /* ID  */
#line 226 "source/parser.y"
            { free(((*yyvaluep).symbol)); }
#line 1742 "source/parser.c"
        break;

    case 101: /* statement_list  */
#line 229 "source/parser.y"
            { osl_statement_free(((*yyvaluep).stmt)); }
#line 1748 "source/parser.c"
        break;

    case 102: /* statement_indented  */
#line 229 "source/parser.y"
            { osl_statement_free(((*yyvaluep).stmt)); }
#line 1754 "source/parser.c"
        break;

    case 104: /* statement  */
#line 229 "source/parser.y"
            { osl_statement_free(((*yyvaluep).stmt)); }
#line 1760 "source/parser.c"
        break;

    case 106: /* labeled_statement  */
#line 229 "source/parser.y"
            { osl_statement_free(((*yyvaluep).stmt)); }
#line 1766 "source/parser.c"
        break;

    case 108: /* compound_statement  */
#line 229 "source/parser.y"
            { osl_statement_free(((*yyvaluep).stmt)); }
#line 1772 "source/parser.c"
        break;

    case 109: /* selection_else_statement  */
#line 229 "source/parser.y"
            { osl_statement_free(((*yyvaluep).stmt)); }
#line 1778 "source/parser.c"
        break;

    case 111: /* selection_statement  */
#line 229 "source/parser.y"
            { osl_statement_free(((*yyvaluep).stmt)); }
#line 1784 "source/parser.c"
        break;

    case 114: /* iteration_statement  */
#line 229 "source/parser.y"
            { osl_statement_free(((*yyvaluep).stmt)); }
#line 1790 "source/parser.c"
        break;

    case 118: /* loop_initialization_list  */
#line 228 "source/parser.y"
            { osl_relation_list_free(((*yyvaluep).list)); }
#line 1796 "source/parser.c"
        break;

    case 122: /* loop_condition_list  */
#line 228 "source/parser.y"
            { osl_relation_list_free(((*yyvaluep).list)); }
#line 1802 "source/parser.c"
        break;

    case 126: /* idparent  */
#line 226 "source/parser.y"
            { free(((*yyvaluep).symbol)); }
#line 1808 "source/parser.c"
        break;

    case 128: /* loop_body  */
#line 229 "source/parser.y"
            { osl_statement_free(((*yyvaluep).stmt)); }
#line 1814 "source/parser.c"
        break;

    case 136: /* affine_primary_expression  */
#line 227 "source/parser.y"
            { osl_vector_free(((*yyvaluep).affex)); }
#line 1820 "source/parser.c"
        break;

    case 137: /* affine_unary_expression  */
#line 227 "source/parser.y"
            { osl_vector_free(((*yyvaluep).affex)); }
#line 1826 "source/parser.c"
        break;

    case 138: /* affine_multiplicative_expression  */
#line 227 "source/parser.y"
            { osl_vector_free(((*yyvaluep).affex)); }
#line 1832 "source/parser.c"
        break;

    case 139: /* affine_expression  */
#line 227 "source/parser.y"
            { osl_vector_free(((*yyvaluep).affex)); }
#line 1838 "source/parser.c"
        break;

    case 140: /* affine_ceildfloord_expression  */
#line 227 "source/parser.y"
            { osl_vector_free(((*yyvaluep).affex)); }
#line 1844 "source/parser.c"
        break;

    case 142: /* affine_ceild_expression  */
#line 227 "source/parser.y"
            { osl_vector_free(((*yyvaluep).affex)); }
#line 1850 "source/parser.c"
        break;

    case 143: /* affine_floord_expression  */
#line 227 "source/parser.y"
            { osl_vector_free(((*yyvaluep).affex)); }
#line 1856 "source/parser.c"
        break;

    case 144: /* id_or_clan_keyword  */
#line 226 "source/parser.y"
            { free(((*yyvaluep).symbol)); }
#line 1862 "source/parser.c"
        break;

    case 145: /* primary_expression  */
#line 228 "source/parser.y"
            { osl_relation_list_free(((*yyvaluep).list)); }
#line 1868 "source/parser.c"
        break;

    case 146: /* postfix_expression  */
#line 228 "source/parser.y"
            { osl_relation_list_free(((*yyvaluep).list)); }
#line 1874 "source/parser.c"
        break;

    case 148: /* argument_expression_list  */
#line 228 "source/parser.y"
            { osl_relation_list_free(((*yyvaluep).list)); }
#line 1880 "source/parser.c"
        break;

    case 149: /* unary_expression  */
#line 228 "source/parser.y"
            { osl_relation_list_free(((*yyvaluep).list)); }
#line 1886 "source/parser.c"
        break;

    case 152: /* cast_expression  */
#line 228 "source/parser.y"
            { osl_relation_list_free(((*yyvaluep).list)); }
#line 1892 "source/parser.c"
        break;

    case 153: /* multiplicative_expression  */
#line 228 "source/parser.y"
            { osl_relation_list_free(((*yyvaluep).list)); }
#line 1898 "source/parser.c"
        break;

    case 154: /* additive_expression  */
#line 228 "source/parser.y"
            { osl_relation_list_free(((*yyvaluep).list)); }
#line 1904 "source/parser.c"
        break;

    case 155: /* shift_expression  */
#line 228 "source/parser.y"
            { osl_relation_list_free(((*yyvaluep).list)); }
#line 1910 "source/parser.c"
        break;

    case 156: /* relational_expression  */
#line 228 "source/parser.y"
            { osl_relation_list_free(((*yyvaluep).list)); }
#line 1916 "source/parser.c"
        break;

    case 157: /* equality_expression  */
#line 228 "source/parser.y"
            { osl_relation_list_free(((*yyvaluep).list)); }
#line 1922 "source/parser.c"
        break;

    case 158: /* and_expression  */
#line 228 "source/parser.y"
            { osl_relation_list_free(((*yyvaluep).list)); }
#line 1928 "source/parser.c"
        break;

    case 159: /* exclusive_or_expression  */
#line 228 "source/parser.y"
            { osl_relation_list_free(((*yyvaluep).list)); }
#line 1934 "source/parser.c"
        break;

    case 160: /* inclusive_or_expression  */
#line 228 "source/parser.y"
            { osl_relation_list_free(((*yyvaluep).list)); }
#line 1940 "source/parser.c"
        break;

    case 161: /* logical_and_expression  */
#line 228 "source/parser.y"
            { osl_relation_list_free(((*yyvaluep).list)); }
#line 1946 "source/parser.c"
        break;

    case 162: /* logical_or_expression  */
#line 228 "source/parser.y"
            { osl_relation_list_free(((*yyvaluep).list)); }
#line 1952 "source/parser.c"
        break;

    case 163: /* conditional_expression  */
#line 228 "source/parser.y"
            { osl_relation_list_free(((*yyvaluep).list)); }
#line 1958 "source/parser.c"
        break;

    case 164: /* assignment_expression  */
#line 228 "source/parser.y"
            { osl_relation_list_free(((*yyvaluep).list)); }
#line 1964 "source/parser.c"
        break;

    case 167: /* expression  */
#line 228 "source/parser.y"
            { osl_relation_list_free(((*yyvaluep).list)); }
#line 1970 "source/parser.c"
        break;

    case 168: /* expression_statement  */
#line 229 "source/parser.y"
            { osl_statement_free(((*yyvaluep).stmt)); }
#line 1976 "source/parser.c"
        break;

      default:
        break;
    }
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}




/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    yy_state_fast_t yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss;
    yy_state_t *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYPTRDIFF_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYPTRDIFF_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    goto yyexhaustedlab;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
# undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 2:
#line 241 "source/parser.y"
                     { CLAN_debug("rule scop_list.1: scop"); }
#line 2246 "source/parser.c"
    break;

  case 3:
#line 242 "source/parser.y"
                     { CLAN_debug("rule scop_list.2: scop_list scop"); }
#line 2252 "source/parser.c"
    break;

  case 4:
#line 243 "source/parser.y"
                     { CLAN_debug("rule scop_list.3: scop_list IGNORE"); }
#line 2258 "source/parser.c"
    break;

  case 5:
#line 244 "source/parser.y"
                     { CLAN_debug("rule scop_list.4: IGNORE"); }
#line 2264 "source/parser.c"
    break;

  case 6:
#line 251 "source/parser.y"
    {
      int nb_parameters;
      osl_scop_p scop;
      osl_generic_p arrays;

      CLAN_debug("rule scop.1: statement_list IGNORE");
      scop = osl_scop_malloc();
      CLAN_strdup(scop->language, "C");

      // Build the SCoP context.
      nb_parameters = clan_symbol_nb_of_type(parser_symbol,
          CLAN_TYPE_PARAMETER);
      scop->parameters = clan_symbol_to_strings(parser_symbol,
          CLAN_TYPE_PARAMETER);
      scop->context = clan_relation_build_context(nb_parameters,
                                                  parser_options);
      
      // Set the statements.
      scop->statement = (yyvsp[-1].stmt);

      // Compact the SCoP relations.
      if (CLAN_DEBUG) {
	CLAN_debug("SCoP before compaction:");
	osl_scop_dump(stderr, scop);
      }
      clan_scop_compact(scop);
      if (CLAN_DEBUG) {
	CLAN_debug("SCoP after compaction:");
	osl_scop_dump(stderr, scop);
      }

      // Simplify the SCoP iteration domains.
      if (!parser_options->nosimplify)
        clan_scop_simplify(scop);

      // Add extensions.
      scop->registry = osl_interface_get_default_registry();
      clan_scop_generate_scatnames(scop);
      arrays = clan_symbol_to_arrays(parser_symbol);
      osl_generic_add(&scop->extension, arrays);
      clan_scop_generate_coordinates(scop, parser_options->name);
      clan_scop_generate_clay(scop, scanner_clay);

      // Add the SCoP to parser_scop and prepare the state for the next SCoP.
      osl_scop_add(&parser_scop, scop);
      clan_symbol_free(parser_symbol);
      clan_parser_state_initialize(parser_options);
      CLAN_debug_call(osl_scop_dump(stderr, scop));
    }
#line 2318 "source/parser.c"
    break;

  case 7:
#line 306 "source/parser.y"
                             { (yyval.stmt) = (yyvsp[0].stmt); }
#line 2324 "source/parser.c"
    break;

  case 8:
#line 308 "source/parser.y"
                             { (yyval.stmt) = (yyvsp[-1].stmt); osl_statement_add(&(yyval.stmt), (yyvsp[0].stmt)); }
#line 2330 "source/parser.c"
    break;

  case 9:
#line 315 "source/parser.y"
    { 
      if (parser_indent == CLAN_UNDEFINED)
        parser_indent = scanner_column_LALR - 1;
    }
#line 2339 "source/parser.c"
    break;

  case 10:
#line 320 "source/parser.y"
    {
      (yyval.stmt) = (yyvsp[0].stmt);
    }
#line 2347 "source/parser.c"
    break;

  case 11:
#line 329 "source/parser.y"
                             { (yyval.stmt) = (yyvsp[0].stmt); }
#line 2353 "source/parser.c"
    break;

  case 12:
#line 330 "source/parser.y"
                             { (yyval.stmt) = (yyvsp[0].stmt); }
#line 2359 "source/parser.c"
    break;

  case 13:
#line 331 "source/parser.y"
                             { (yyval.stmt) = (yyvsp[0].stmt); }
#line 2365 "source/parser.c"
    break;

  case 14:
#line 332 "source/parser.y"
                             { (yyval.stmt) = (yyvsp[0].stmt); }
#line 2371 "source/parser.c"
    break;

  case 15:
#line 333 "source/parser.y"
    {
      if (parser_options->autoscop && !parser_autoscop && !parser_loop_depth) {
        parser_line_start = scanner_line;
        parser_column_start = scanner_column_LALR;
        parser_autoscop = CLAN_TRUE;
        // Reinitialize the symbol table.
        clan_symbol_free(parser_symbol);
        parser_symbol = NULL;
        if (CLAN_DEBUG)
          fprintf(stderr, "Autoscop start: line %3d column %3d\n",
                  parser_line_start, parser_column_start);
      }
    }
#line 2389 "source/parser.c"
    break;

  case 16:
#line 347 "source/parser.y"
    {
      (yyval.stmt) = (yyvsp[0].stmt);
      if (parser_options->autoscop && parser_autoscop && !parser_loop_depth) {
        parser_line_end = scanner_line;
        parser_column_end = scanner_column;
        if (CLAN_DEBUG)
          fprintf(stderr, "Autoscop found: line %3d column %3d\n",
                  parser_line_end, parser_column_end);
      }
    }
#line 2404 "source/parser.c"
    break;

  case 17:
#line 362 "source/parser.y"
    {
      int i;
      clan_domain_p labeled_domain;
      osl_relation_list_p labeled_constraints;

      CLAN_debug("labeled_statement.1.1: <int> : ...");
     
      if (parser_stack == NULL)
        printf("NULL stack, label %d\n", (yyvsp[-1].value));
      if (parser_stack->constraints == NULL)
        printf("NULL constraints\n");
      if (((yyvsp[-1].value) < 0) ||
	  ((yyvsp[-1].value) >= clan_relation_list_nb_elements(parser_stack->constraints))) {
	yyerror("label out of range");
        YYABORT;
      }

      labeled_domain = clan_domain_malloc();
      labeled_domain->constraints = osl_relation_list_malloc();
      labeled_constraints = parser_stack->constraints;
      for (i = 0; i < (yyvsp[-1].value); i++)
	labeled_constraints = labeled_constraints->next;
      labeled_domain->constraints->elt =
	  osl_relation_clone(labeled_constraints->elt);

      clan_domain_push(&parser_stack, labeled_domain);
      parser_xfor_labels[parser_xfor_nb_nests] = (yyvsp[-1].value);
      parser_xfor_depths[parser_xfor_nb_nests + 1] = 0;
      parser_xfor_nb_nests++;
    }
#line 2439 "source/parser.c"
    break;

  case 18:
#line 393 "source/parser.y"
    {
      clan_domain_drop(&parser_stack);
      parser_xfor_nb_nests--;
      parser_xfor_labels[parser_xfor_nb_nests] = CLAN_UNDEFINED;
      (yyval.stmt) = (yyvsp[0].stmt);
      CLAN_debug("labeled_statement.1.2: ... <stmt>");
    }
#line 2451 "source/parser.c"
    break;

  case 19:
#line 406 "source/parser.y"
                             { (yyval.stmt) = NULL; }
#line 2457 "source/parser.c"
    break;

  case 20:
#line 407 "source/parser.y"
                             { (yyval.stmt) = (yyvsp[-1].stmt); }
#line 2463 "source/parser.c"
    break;

  case 21:
#line 418 "source/parser.y"
    {
      if (!parser_valid_else[parser_if_depth]) {
	yyerror("unsupported negation of a condition involving a modulo");
	YYABORT;
      }
    }
#line 2474 "source/parser.c"
    break;

  case 22:
#line 425 "source/parser.y"
    {
      CLAN_debug("rule selection_else_statement.1: else <stmt>");
      (yyval.stmt) = (yyvsp[0].stmt);
      CLAN_debug_call(osl_statement_dump(stderr, (yyval.stmt)));
    }
#line 2484 "source/parser.c"
    break;

  case 23:
#line 431 "source/parser.y"
    {
      CLAN_debug("rule selection_else_statement.2: <void>");
      (yyval.stmt) = NULL;
    }
#line 2493 "source/parser.c"
    break;

  case 24:
#line 440 "source/parser.y"
    {
      CLAN_debug("rule selection_statement.1.1: if ( condition ) ...");
      clan_domain_dup(&parser_stack);
      clan_domain_and(parser_stack, (yyvsp[-1].setex));
      parser_if_depth++;
      if ((parser_loop_depth + parser_if_depth) > CLAN_MAX_DEPTH)
	CLAN_error("CLAN_MAX_DEPTH reached, recompile with a higher value");
    }
#line 2506 "source/parser.c"
    break;

  case 25:
#line 449 "source/parser.y"
    {
      osl_relation_p not_if;
      
      CLAN_debug("rule selection_statement.1.2: if ( condition ) <stmt> ...");
      clan_domain_drop(&parser_stack);
      clan_domain_dup(&parser_stack);
      if (!clan_relation_existential((yyvsp[-3].setex))) {
	not_if = clan_relation_not((yyvsp[-3].setex));
	clan_domain_and(parser_stack, not_if);
	osl_relation_free(not_if);
	parser_valid_else[parser_if_depth] = 1;
      } else {
	parser_valid_else[parser_if_depth] = 0;
      }
      osl_relation_free((yyvsp[-3].setex));
    }
#line 2527 "source/parser.c"
    break;

  case 26:
#line 466 "source/parser.y"
    {
      CLAN_debug("rule selection_statement.1.3: if ( condition ) <stmt>"
	         "[else <stmt>]");
      clan_domain_drop(&parser_stack);
      (yyval.stmt) = (yyvsp[-2].stmt);
      osl_statement_add(&(yyval.stmt), (yyvsp[0].stmt));
      parser_if_depth--;
      parser_nb_local_dims[parser_loop_depth + parser_if_depth] = 0;
      CLAN_debug_call(osl_statement_dump(stderr, (yyval.stmt)));
    }
#line 2542 "source/parser.c"
    break;

  case 27:
#line 481 "source/parser.y"
    {
      CLAN_debug("rule iteration_statement.1.1: xfor ( init cond stride ) ...");
      parser_xfor_labels[parser_loop_depth] = CLAN_UNDEFINED;
       
      // Check loop bounds and stride consistency and reset sanity sentinels.
      if (!clan_parser_is_loop_sane((yyvsp[-3].list), (yyvsp[-2].list), (yyvsp[-1].vecint)))
        YYABORT;

      // Check that either an xfor loop is the first one or have the same
      // number of indices than the previous one.
      if ((clan_relation_list_nb_elements(parser_stack->constraints) != 1) &&
	  (clan_relation_list_nb_elements(parser_stack->constraints) !=
	   clan_relation_list_nb_elements((yyvsp[-3].list)))) {
	yyerror("consecutive xfor loops without the same number of indices");
	osl_relation_list_free((yyvsp[-3].list));
        osl_relation_list_free((yyvsp[-2].list));
	free((yyvsp[-1].vecint));
        YYABORT;
      }

      // Add the constraints contributed by the xfor loop to the domain stack.
      clan_domain_dup(&parser_stack);
      clan_domain_xfor(parser_stack, parser_loop_depth + 1, parser_symbol,
	               (yyvsp[-3].list), (yyvsp[-2].list), (yyvsp[-1].vecint), parser_options);

      clan_parser_increment_loop_depth();
      parser_xfor_depths[parser_xfor_nb_nests]++;
      parser_xfor_index = 0;
      osl_relation_list_free((yyvsp[-3].list));
      osl_relation_list_free((yyvsp[-2].list));
      (yyvsp[-3].list) = NULL; // To avoid conflicts with the destructor TODO: avoid that.
      (yyvsp[-2].list) = NULL;
      parser_scattering[2*parser_loop_depth-1] = ((yyvsp[-1].vecint)[0] > 0) ? 1 : -1;
      parser_scattering[2*parser_loop_depth] = 0;
      free((yyvsp[-1].vecint));
    }
#line 2583 "source/parser.c"
    break;

  case 28:
#line 518 "source/parser.y"
    {
      CLAN_debug("rule iteration_statement.1.2: xfor ( init cond stride ) "
	         "body");
      parser_xfor_depths[parser_xfor_nb_nests]--;
      (yyval.stmt) = (yyvsp[0].stmt);
      CLAN_debug_call(osl_statement_dump(stderr, (yyval.stmt)));
    }
#line 2595 "source/parser.c"
    break;

  case 29:
#line 527 "source/parser.y"
    {
      CLAN_debug("rule iteration_statement.2.1: for ( init cond stride ) ...");
      parser_xfor_labels[parser_loop_depth] = 0;
     
      // Check there is only one element in each list
      if (parser_xfor_index != 1) {
	yyerror("unsupported element list in a for loop");
	osl_relation_list_free((yyvsp[-3].list));
        osl_relation_list_free((yyvsp[-2].list));
	free((yyvsp[-1].vecint));
        YYABORT;
      }

      // Check loop bounds and stride consistency and reset sanity sentinels.
      if (!clan_parser_is_loop_sane((yyvsp[-3].list), (yyvsp[-2].list), (yyvsp[-1].vecint)))
        YYABORT;

      // Add the constraints contributed by the for loop to the domain stack.
      clan_domain_dup(&parser_stack);
      clan_domain_for(parser_stack, parser_loop_depth + 1, parser_symbol,
	              (yyvsp[-3].list)->elt, (yyvsp[-2].list)->elt, (yyvsp[-1].vecint)[0], parser_options);

      clan_parser_increment_loop_depth();
      parser_xfor_index = 0;
      osl_relation_list_free((yyvsp[-3].list));
      osl_relation_list_free((yyvsp[-2].list));
      (yyvsp[-3].list) = NULL; // To avoid conflicts with the destructor TODO: avoid that.
      (yyvsp[-2].list) = NULL;
      parser_scattering[2*parser_loop_depth-1] = ((yyvsp[-1].vecint)[0] > 0) ? 1 : -1;
      parser_scattering[2*parser_loop_depth] = 0;
      free((yyvsp[-1].vecint));
    }
#line 2632 "source/parser.c"
    break;

  case 30:
#line 560 "source/parser.y"
    {
      CLAN_debug("rule iteration_statement.2.2: for ( init cond stride ) "
	         "body");
      (yyval.stmt) = (yyvsp[0].stmt);
      CLAN_debug_call(osl_statement_dump(stderr, (yyval.stmt)));
    }
#line 2643 "source/parser.c"
    break;

  case 31:
#line 567 "source/parser.y"
    {
      osl_vector_p   iterator_term;
      osl_relation_p iterator_relation;

      CLAN_debug("rule iteration_statement.3.1: loop_infinite ...");
      if (!clan_symbol_new_iterator(&parser_symbol, parser_iterators,
	                            "clan_infinite_loop", parser_loop_depth))
	YYABORT;

      parser_xfor_labels[parser_loop_depth] = 0;
      clan_parser_increment_loop_depth();
      
      // Generate the constraint clan_infinite_loop >= 0.
      iterator_term = clan_vector_term(parser_symbol, 0, NULL,
                                       parser_options->precision);
      osl_int_set_si(parser_options->precision,
                     &iterator_term->v[parser_loop_depth], 1); 
      osl_int_set_si(parser_options->precision, &iterator_term->v[0], 1); 
      iterator_relation = osl_relation_from_vector(iterator_term);
      
      // Add it to the domain stack.
      clan_domain_dup(&parser_stack);
      clan_domain_and(parser_stack, iterator_relation);
      osl_vector_free(iterator_term);
      osl_relation_free(iterator_relation);
      parser_scattering[2*parser_loop_depth-1] = 1;
      parser_scattering[2*parser_loop_depth] = 0;
    }
#line 2676 "source/parser.c"
    break;

  case 32:
#line 596 "source/parser.y"
    {
      CLAN_debug("rule iteration_statement.3.2: loop_infinite body");
      (yyval.stmt) = (yyvsp[0].stmt);
      CLAN_debug_call(osl_statement_dump(stderr, (yyval.stmt)));
    }
#line 2686 "source/parser.c"
    break;

  case 33:
#line 606 "source/parser.y"
    {
      osl_relation_list_p new = osl_relation_list_malloc();
      CLAN_debug("rule initialization_list.1: initialization , "
	         "initialization_list");
      new->elt = (yyvsp[-2].setex);
      osl_relation_list_push(&(yyvsp[0].list), new);
      (yyval.list) = (yyvsp[0].list);
    }
#line 2699 "source/parser.c"
    break;

  case 34:
#line 615 "source/parser.y"
    {
      CLAN_debug("rule initialization_list.2: initialization ;");
      parser_xfor_index = 0;
      (yyval.list) = osl_relation_list_malloc();
      (yyval.list)->elt = (yyvsp[-1].setex);
    }
#line 2710 "source/parser.c"
    break;

  case 35:
#line 626 "source/parser.y"
    {
      if (!clan_symbol_new_iterator(&parser_symbol, parser_iterators, (yyvsp[0].symbol),
	                            parser_loop_depth))
	YYABORT;
    }
#line 2720 "source/parser.c"
    break;

  case 36:
#line 632 "source/parser.y"
    {
      CLAN_debug("rule initialization: ID = <setex>");
      parser_xfor_index++;
      free((yyvsp[-3].symbol));
      (yyval.setex) = (yyvsp[0].setex);
      CLAN_debug_call(osl_relation_dump(stderr, (yyval.setex)));
    }
#line 2732 "source/parser.c"
    break;

  case 39:
#line 650 "source/parser.y"
    {
      osl_relation_list_p new = osl_relation_list_malloc();
      new->elt = (yyvsp[-2].setex);
      osl_relation_list_push(&(yyvsp[0].list), new);
      (yyval.list) = (yyvsp[0].list);
    }
#line 2743 "source/parser.c"
    break;

  case 40:
#line 657 "source/parser.y"
    {
      parser_xfor_index = 0;
      (yyval.list) = osl_relation_list_malloc();
      (yyval.list)->elt = (yyvsp[-1].setex);
    }
#line 2753 "source/parser.c"
    break;

  case 41:
#line 667 "source/parser.y"
    {
      CLAN_debug("rule condition.1: <setex>");
      parser_xfor_index++;
      (yyval.setex) = (yyvsp[0].setex);
      CLAN_debug_call(osl_relation_dump(stderr, (yyval.setex)));
    }
#line 2764 "source/parser.c"
    break;

  case 42:
#line 678 "source/parser.y"
    {
      int i;
      (yyval.vecint) = malloc((parser_xfor_index) * sizeof(int));
      for (i = 0; i < parser_xfor_index - 1; i++)
        (yyval.vecint)[i + 1] = (yyvsp[0].vecint)[i];
      free((yyvsp[0].vecint));
      (yyval.vecint)[0] = (yyvsp[-2].value);
    }
#line 2777 "source/parser.c"
    break;

  case 43:
#line 687 "source/parser.y"
    {
      (yyval.vecint) = malloc(sizeof(int));
      (yyval.vecint)[0] = (yyvsp[0].value);
    }
#line 2786 "source/parser.c"
    break;

  case 44:
#line 701 "source/parser.y"
                                { parser_xfor_index++; (yyval.value) =  1;  free((yyvsp[-1].symbol)); }
#line 2792 "source/parser.c"
    break;

  case 45:
#line 702 "source/parser.y"
                                { parser_xfor_index++; (yyval.value) = -1;  free((yyvsp[-1].symbol)); }
#line 2798 "source/parser.c"
    break;

  case 46:
#line 703 "source/parser.y"
                                { parser_xfor_index++; (yyval.value) =  1;  free((yyvsp[0].symbol)); }
#line 2804 "source/parser.c"
    break;

  case 47:
#line 704 "source/parser.y"
                                { parser_xfor_index++; (yyval.value) = -1;  free((yyvsp[0].symbol)); }
#line 2810 "source/parser.c"
    break;

  case 48:
#line 706 "source/parser.y"
    { parser_xfor_index++; (yyval.value) =  (yyvsp[0].value); free((yyvsp[-4].symbol)); free((yyvsp[-2].symbol)); }
#line 2816 "source/parser.c"
    break;

  case 49:
#line 708 "source/parser.y"
    { parser_xfor_index++; (yyval.value) = -(yyvsp[0].value); free((yyvsp[-4].symbol)); free((yyvsp[-2].symbol)); }
#line 2822 "source/parser.c"
    break;

  case 50:
#line 709 "source/parser.y"
                                { parser_xfor_index++; (yyval.value) =  (yyvsp[0].value); free((yyvsp[-2].symbol)); }
#line 2828 "source/parser.c"
    break;

  case 51:
#line 710 "source/parser.y"
                                { parser_xfor_index++; (yyval.value) = -(yyvsp[0].value); free((yyvsp[-2].symbol)); }
#line 2834 "source/parser.c"
    break;

  case 52:
#line 714 "source/parser.y"
       { (yyval.symbol) = (yyvsp[0].symbol); }
#line 2840 "source/parser.c"
    break;

  case 53:
#line 716 "source/parser.y"
    { (yyval.symbol) = (yyvsp[-1].symbol); }
#line 2846 "source/parser.c"
    break;

  case 56:
#line 727 "source/parser.y"
    {
      CLAN_debug("rule loop_body.1: <stmt>");
      parser_loop_depth--;
      clan_symbol_free(parser_iterators[parser_loop_depth]);
      parser_iterators[parser_loop_depth] = NULL;
      clan_domain_drop(&parser_stack);
      (yyval.stmt) = (yyvsp[0].stmt);
      parser_scattering[2*parser_loop_depth]++;
      parser_nb_local_dims[parser_loop_depth + parser_if_depth] = 0;
      CLAN_debug_call(osl_statement_dump(stderr, (yyval.stmt)));
    }
#line 2862 "source/parser.c"
    break;

  case 57:
#line 748 "source/parser.y"
    {
      CLAN_debug("rule affine_minmax_expression.1: <affex>");
      (yyval.setex) = osl_relation_from_vector((yyvsp[0].affex));
      osl_vector_free((yyvsp[0].affex));
      CLAN_debug_call(osl_relation_dump(stderr, (yyval.setex)));
    }
#line 2873 "source/parser.c"
    break;

  case 58:
#line 755 "source/parser.y"
    {
      CLAN_debug("rule affine_minmax_expression.2: "
                 "MAX (affine_minmaxexpression , affine_minmax_expression )");
      (yyval.setex) = osl_relation_concat_constraints((yyvsp[-3].setex), (yyvsp[-1].setex));
      osl_relation_free((yyvsp[-3].setex));
      osl_relation_free((yyvsp[-1].setex));
      CLAN_debug_call(osl_relation_dump(stderr, (yyval.setex)));
    }
#line 2886 "source/parser.c"
    break;

  case 59:
#line 767 "source/parser.y"
        { parser_min[parser_xfor_index] = 1; }
#line 2892 "source/parser.c"
    break;

  case 60:
#line 768 "source/parser.y"
        { parser_max[parser_xfor_index] = 1; }
#line 2898 "source/parser.c"
    break;

  case 61:
#line 781 "source/parser.y"
    {
      CLAN_debug("rule affine_min_expression.1: <affex>");
      (yyval.setex) = osl_relation_from_vector((yyvsp[0].affex));
      osl_vector_free((yyvsp[0].affex));
      CLAN_debug_call(osl_relation_dump(stderr, (yyval.setex)));
    }
#line 2909 "source/parser.c"
    break;

  case 62:
#line 791 "source/parser.y"
    {
      CLAN_debug("rule affine_min_expression.2: "
                 "MIN ( affine_min_expression , affine_min_expresssion");
      (yyval.setex) = osl_relation_concat_constraints((yyvsp[-3].setex), (yyvsp[-1].setex));
      osl_relation_free((yyvsp[-3].setex));
      osl_relation_free((yyvsp[-1].setex));
      CLAN_debug_call(osl_relation_dump(stderr, (yyval.setex)));
    }
#line 2922 "source/parser.c"
    break;

  case 63:
#line 811 "source/parser.y"
    {
      CLAN_debug("rule affine_max_expression.1: <affex>");
      (yyval.setex) = osl_relation_from_vector((yyvsp[0].affex));
      osl_vector_free((yyvsp[0].affex));
      CLAN_debug_call(osl_relation_dump(stderr, (yyval.setex)));
    }
#line 2933 "source/parser.c"
    break;

  case 64:
#line 821 "source/parser.y"
    {
      CLAN_debug("rule affine_max_expression.2: "
                 "MAX ( affine_max_expression , affine_max_expression )");
      (yyval.setex) = osl_relation_concat_constraints((yyvsp[-3].setex), (yyvsp[-1].setex));
      osl_relation_free((yyvsp[-3].setex));
      osl_relation_free((yyvsp[-1].setex));
      CLAN_debug_call(osl_relation_dump(stderr, (yyval.setex)));
    }
#line 2946 "source/parser.c"
    break;

  case 65:
#line 844 "source/parser.y"
    {
      CLAN_debug("rule affine_relation.1: max_affex < min_affex");
      (yyval.setex) = clan_relation_greater((yyvsp[0].setex), (yyvsp[-2].setex), 1);
      osl_relation_free((yyvsp[-2].setex));
      osl_relation_free((yyvsp[0].setex));
      CLAN_debug_call(osl_relation_dump(stderr, (yyval.setex)));
    }
#line 2958 "source/parser.c"
    break;

  case 66:
#line 855 "source/parser.y"
    {
      CLAN_debug("rule affine_relation.2: min_affex > max_affex");
      (yyval.setex) = clan_relation_greater((yyvsp[-2].setex), (yyvsp[0].setex), 1);
      osl_relation_free((yyvsp[-2].setex));
      osl_relation_free((yyvsp[0].setex));
      CLAN_debug_call(osl_relation_dump(stderr, (yyval.setex)));
    }
#line 2970 "source/parser.c"
    break;

  case 67:
#line 866 "source/parser.y"
    {
      CLAN_debug("rule affine_relation.3: max_affex <= min_affex");
      (yyval.setex) = clan_relation_greater((yyvsp[0].setex), (yyvsp[-2].setex), 0);
      osl_relation_free((yyvsp[-2].setex));
      osl_relation_free((yyvsp[0].setex));
      CLAN_debug_call(osl_relation_dump(stderr, (yyval.setex)));
    }
#line 2982 "source/parser.c"
    break;

  case 68:
#line 877 "source/parser.y"
    {
      CLAN_debug("rule affine_relation.4: min_affex >= max_affex");
      (yyval.setex) = clan_relation_greater((yyvsp[-2].setex), (yyvsp[0].setex), 0);
      osl_relation_free((yyvsp[-2].setex));
      osl_relation_free((yyvsp[0].setex));
      CLAN_debug_call(osl_relation_dump(stderr, (yyval.setex)));
    }
#line 2994 "source/parser.c"
    break;

  case 69:
#line 888 "source/parser.y"
    {
      // a==b translates to a-b==0.
      osl_vector_p res;

      CLAN_debug("rule affine_relation.5: <affex> == <affex>");
      // Warning: cases like ceild(M,32) == ceild(N,32) are not handled.
      // Assert if we encounter such a case.
      assert ((osl_int_zero(parser_options->precision, (yyvsp[-2].affex)->v[0]) ||
	       osl_int_one(parser_options->precision,  (yyvsp[-2].affex)->v[0])) &&
	      (osl_int_zero(parser_options->precision, (yyvsp[0].affex)->v[0]) ||
	       osl_int_one(parser_options->precision,  (yyvsp[0].affex)->v[0])));
      res = osl_vector_sub((yyvsp[-2].affex), (yyvsp[0].affex));
      osl_vector_tag_equality(res);
      (yyval.setex) = osl_relation_from_vector(res);
      osl_vector_free(res);
      osl_vector_free((yyvsp[-2].affex));
      osl_vector_free((yyvsp[0].affex));
      CLAN_debug_call(osl_relation_dump(stderr, (yyval.setex)));
    }
#line 3018 "source/parser.c"
    break;

  case 70:
#line 911 "source/parser.y"
    {
      CLAN_debug("rule affine_relation.6: ( condition )");
      (yyval.setex) = (yyvsp[-1].setex);
      CLAN_debug_call(osl_relation_dump(stderr, (yyval.setex)));
    }
#line 3028 "source/parser.c"
    break;

  case 71:
#line 920 "source/parser.y"
    {
      CLAN_debug("rule affine_relation.7: ! ( condition )");
      if (clan_relation_existential((yyvsp[-1].setex))) {
        osl_relation_free((yyvsp[-1].setex));
	yyerror("unsupported negation of a condition involving a modulo");
	YYABORT;
      }
      (yyval.setex) = clan_relation_not((yyvsp[-1].setex));
      osl_relation_free((yyvsp[-1].setex));
      CLAN_debug_call(osl_relation_dump(stderr, (yyval.setex)));
    }
#line 3044 "source/parser.c"
    break;

  case 72:
#line 935 "source/parser.y"
    {
      CLAN_debug("rule affine_relation.8: "
	         "affine_expression %% INTEGER == INTEGER");
      osl_int_set_si(parser_options->precision,
                     &((yyvsp[-4].affex)->v[CLAN_MAX_DEPTH + 1 + clan_parser_nb_ld()]), -(yyvsp[-2].value));
      osl_int_add_si(parser_options->precision,
	             &((yyvsp[-4].affex)->v[(yyvsp[-4].affex)->size - 1]), (yyvsp[-4].affex)->v[(yyvsp[-4].affex)->size - 1], -(yyvsp[0].value));
      clan_parser_add_ld();
      (yyval.setex) = osl_relation_from_vector((yyvsp[-4].affex));
      osl_vector_free((yyvsp[-4].affex));
      CLAN_debug_call(osl_relation_dump(stderr, (yyval.setex)));
    }
#line 3061 "source/parser.c"
    break;

  case 73:
#line 952 "source/parser.y"
    {
      CLAN_debug("rule affine_logical_and_expression.1: affine_relation");
      (yyval.setex) = (yyvsp[0].setex);
      CLAN_debug_call(osl_relation_dump(stderr, (yyval.setex)));
    }
#line 3071 "source/parser.c"
    break;

  case 74:
#line 958 "source/parser.y"
    {
      CLAN_debug("rule affine_logical_and_expression.2: "
	         "affine_logical_and_expression && affine_relation");
      clan_relation_and((yyvsp[-2].setex), (yyvsp[0].setex));
      (yyval.setex) = (yyvsp[-2].setex);
      osl_relation_free((yyvsp[0].setex));
      CLAN_debug_call(osl_relation_dump(stderr, (yyval.setex)));
    }
#line 3084 "source/parser.c"
    break;

  case 75:
#line 971 "source/parser.y"
    {
      CLAN_debug("rule affine_condition.1: affine_logical_and_expression");
      (yyval.setex) = (yyvsp[0].setex);
      CLAN_debug_call(osl_relation_dump(stderr, (yyval.setex)));
    }
#line 3094 "source/parser.c"
    break;

  case 76:
#line 977 "source/parser.y"
    {
      CLAN_debug("rule affine_condition.2: "
	         "affine_condition || affine_logical_and_expression");
      osl_relation_add(&(yyvsp[-2].setex), (yyvsp[0].setex));
      (yyval.setex) = (yyvsp[-2].setex);
      CLAN_debug_call(osl_relation_dump(stderr, (yyval.setex)));
    }
#line 3106 "source/parser.c"
    break;

  case 77:
#line 989 "source/parser.y"
    {
      clan_symbol_p id;

      CLAN_debug("rule affine_primary_expression.1: id");
      id = clan_symbol_add(&parser_symbol, (yyvsp[0].symbol), CLAN_UNDEFINED);
      // An id in an affex can be either an iterator or a parameter. If it is
      // an unknown (embeds read-only variables), it is updated to a parameter.
      if (id->type == CLAN_UNDEFINED) {
        if ((parser_nb_parameters + 1) > CLAN_MAX_PARAMETERS)
	        CLAN_error("CLAN_MAX_PARAMETERS reached,"
                             "recompile with a higher value");
        id->type = CLAN_TYPE_PARAMETER;
        id->rank = ++parser_nb_parameters;
      }

      if ((id->type != CLAN_TYPE_ITERATOR) &&
          (id->type != CLAN_TYPE_PARAMETER)) {
        free((yyvsp[0].symbol));
	if (id->type == CLAN_TYPE_ARRAY)
	  yyerror("variable or array reference in an affine expression");
	else
          yyerror("function call in an affine expression");
	YYABORT;
      }
      
      (yyval.affex) = clan_vector_term(parser_symbol, 1, (yyvsp[0].symbol), parser_options->precision);
      free((yyvsp[0].symbol));
      CLAN_debug_call(osl_vector_dump(stderr, (yyval.affex)));
    }
#line 3140 "source/parser.c"
    break;

  case 78:
#line 1019 "source/parser.y"
    {
      CLAN_debug("rule affine_primary_expression.2: INTEGER");
      (yyval.affex) = clan_vector_term(parser_symbol, (yyvsp[0].value), NULL, parser_options->precision);
      CLAN_debug_call(osl_vector_dump(stderr, (yyval.affex)));
    }
#line 3150 "source/parser.c"
    break;

  case 79:
#line 1025 "source/parser.y"
    {
      CLAN_debug("rule affine_primary_expression.3: "
                 "affine_additive_expression");
      (yyval.affex) = (yyvsp[-1].affex);
      CLAN_debug_call(osl_vector_dump(stderr, (yyval.affex)));
    }
#line 3161 "source/parser.c"
    break;

  case 80:
#line 1036 "source/parser.y"
    {
      CLAN_debug("rule affine_unary_expression.1: affine_primary_expression");
      (yyval.affex) = (yyvsp[0].affex);
      CLAN_debug_call(osl_vector_dump(stderr, (yyval.affex)));
    }
#line 3171 "source/parser.c"
    break;

  case 81:
#line 1042 "source/parser.y"
    {
      CLAN_debug("rule affine_unary_expression.2: +affine_primary_expression");
      (yyval.affex) = (yyvsp[0].affex);
      CLAN_debug_call(osl_vector_dump(stderr, (yyval.affex)));
    }
#line 3181 "source/parser.c"
    break;

  case 82:
#line 1048 "source/parser.y"
    {
      CLAN_debug("rule affine_unary_expression.2: -affine_primary_expression");
      (yyval.affex) = osl_vector_mul_scalar((yyvsp[0].affex), -1);
      osl_vector_free((yyvsp[0].affex));
      CLAN_debug_call(osl_vector_dump(stderr, (yyval.affex)));
    }
#line 3192 "source/parser.c"
    break;

  case 83:
#line 1059 "source/parser.y"
    { 
      CLAN_debug("rule affine_multiplicative_expression.1: "
                 "affine_unary_expression");
      (yyval.affex) = (yyvsp[0].affex);
      CLAN_debug_call(osl_vector_dump(stderr, (yyval.affex)));
    }
#line 3203 "source/parser.c"
    break;

  case 84:
#line 1066 "source/parser.y"
    {
      int coef;
      
      CLAN_debug("rule affine_multiplicative_expression.2: "
                 "affine_multiplicative_expression * affine_unary_expression");
      if (!osl_vector_is_scalar((yyvsp[-2].affex)) && !osl_vector_is_scalar((yyvsp[0].affex))) {
        osl_vector_free((yyvsp[-2].affex));
        osl_vector_free((yyvsp[0].affex));
        yyerror("non-affine expression");
	YYABORT;
      }

      if (osl_vector_is_scalar((yyvsp[-2].affex))) {
        coef = osl_int_get_si((yyvsp[-2].affex)->precision, (yyvsp[-2].affex)->v[(yyvsp[-2].affex)->size - 1]);
        (yyval.affex) = osl_vector_mul_scalar((yyvsp[0].affex), coef);
      } else {
        coef = osl_int_get_si((yyvsp[0].affex)->precision, (yyvsp[0].affex)->v[(yyvsp[0].affex)->size - 1]);
        (yyval.affex) = osl_vector_mul_scalar((yyvsp[-2].affex), coef);
      }
      osl_vector_free((yyvsp[-2].affex));
      osl_vector_free((yyvsp[0].affex));
      CLAN_debug_call(osl_vector_dump(stderr, (yyval.affex)));
    }
#line 3231 "source/parser.c"
    break;

  case 85:
#line 1090 "source/parser.y"
    {
      int val1, val2;
      
      CLAN_debug("rule affine_multiplicative_expression.3: "
                 "affine_multiplicative_expression / affine_unary_expression");
      if (!osl_vector_is_scalar((yyvsp[-2].affex)) || !osl_vector_is_scalar((yyvsp[0].affex))) {
        osl_vector_free((yyvsp[-2].affex));
        osl_vector_free((yyvsp[0].affex));
        yyerror("non-affine expression");
	YYABORT;
      }
      val1 = osl_int_get_si((yyvsp[-2].affex)->precision, (yyvsp[-2].affex)->v[(yyvsp[-2].affex)->size - 1]);
      val2 = osl_int_get_si((yyvsp[0].affex)->precision, (yyvsp[0].affex)->v[(yyvsp[0].affex)->size - 1]);
      (yyval.affex) = clan_vector_term(parser_symbol, val1 / val2, NULL,
                            parser_options->precision);
      osl_vector_free((yyvsp[-2].affex));
      osl_vector_free((yyvsp[0].affex));
      CLAN_debug_call(osl_vector_dump(stderr, (yyval.affex)));
    }
#line 3255 "source/parser.c"
    break;

  case 86:
#line 1114 "source/parser.y"
    { 
      CLAN_debug("rule affine_expression.1: "
                 "affine_multiplicative_expression");
      (yyval.affex) = (yyvsp[0].affex);
      CLAN_debug_call(osl_vector_dump(stderr, (yyval.affex)));
    }
#line 3266 "source/parser.c"
    break;

  case 87:
#line 1121 "source/parser.y"
    {
      CLAN_debug("rule affine_expression.2: "
          "affine_expression + affine_multiplicative_expression");
      (yyval.affex) = osl_vector_add((yyvsp[-2].affex), (yyvsp[0].affex));
      osl_vector_free((yyvsp[-2].affex));
      osl_vector_free((yyvsp[0].affex));
      CLAN_debug_call(osl_vector_dump(stderr, (yyval.affex)));
    }
#line 3279 "source/parser.c"
    break;

  case 88:
#line 1130 "source/parser.y"
    {
      CLAN_debug("rule affine_expression.3: "
          "affine_expression - affine_multiplicative_expression");
      (yyval.affex) = osl_vector_sub((yyvsp[-2].affex), (yyvsp[0].affex));
      osl_vector_free((yyvsp[-2].affex));
      osl_vector_free((yyvsp[0].affex));
      CLAN_debug_call(osl_vector_dump(stderr, (yyval.affex)));
    }
#line 3292 "source/parser.c"
    break;

  case 89:
#line 1143 "source/parser.y"
    {
      CLAN_debug("affine_ceildloord_expression.1: affine_expression");
      (yyval.affex) = (yyvsp[0].affex);
      CLAN_debug_call(osl_vector_dump(stderr, (yyval.affex)));
    }
#line 3302 "source/parser.c"
    break;

  case 90:
#line 1149 "source/parser.y"
    {
      CLAN_debug("affine_ceildfloord_expression.2: "
                 "ceildfloord ( affine_expression , INTEGER )");
      osl_int_set_si(parser_options->precision, &((yyvsp[-3].affex)->v[0]), (yyvsp[-1].value));
      (yyval.affex) = (yyvsp[-3].affex);
      CLAN_debug_call(osl_vector_dump(stderr, (yyval.affex)));
    }
#line 3314 "source/parser.c"
    break;

  case 91:
#line 1160 "source/parser.y"
           { parser_ceild[parser_xfor_index]  = 1; }
#line 3320 "source/parser.c"
    break;

  case 92:
#line 1161 "source/parser.y"
           { parser_floord[parser_xfor_index] = 1; }
#line 3326 "source/parser.c"
    break;

  case 93:
#line 1167 "source/parser.y"
    {
      CLAN_debug("affine_ceil_expression.1: affine_expression");
      (yyval.affex) = (yyvsp[0].affex);
      CLAN_debug_call(osl_vector_dump(stderr, (yyval.affex)));
    }
#line 3336 "source/parser.c"
    break;

  case 94:
#line 1173 "source/parser.y"
    {
      CLAN_debug("affine_ceil_expression.2: "
                 "CEILD ( affine_expression , INTEGER )");
      osl_int_set_si(parser_options->precision, &((yyvsp[-3].affex)->v[0]), (yyvsp[-1].value));
      (yyval.affex) = (yyvsp[-3].affex);
      CLAN_debug_call(osl_vector_dump(stderr, (yyval.affex)));
    }
#line 3348 "source/parser.c"
    break;

  case 95:
#line 1185 "source/parser.y"
    {
      CLAN_debug("affine_floor_expression.1: affine_expression");
      (yyval.affex) = (yyvsp[0].affex);
      CLAN_debug_call(osl_vector_dump(stderr, (yyval.affex)));
    }
#line 3358 "source/parser.c"
    break;

  case 96:
#line 1191 "source/parser.y"
    {
      CLAN_debug("affine_floor_expression.2: "
                 "FLOORD ( affine_expression , INTEGER )");
      osl_int_set_si(parser_options->precision, &((yyvsp[-3].affex)->v[0]), (yyvsp[-1].value));
      (yyval.affex) = (yyvsp[-3].affex);
      CLAN_debug_call(osl_vector_dump(stderr, (yyval.affex)));
    }
#line 3370 "source/parser.c"
    break;

  case 97:
#line 1201 "source/parser.y"
       { (yyval.symbol) = (yyvsp[0].symbol); }
#line 3376 "source/parser.c"
    break;

  case 98:
#line 1202 "source/parser.y"
        { (yyval.symbol) = strdup("min"); }
#line 3382 "source/parser.c"
    break;

  case 99:
#line 1203 "source/parser.y"
        { (yyval.symbol) = strdup("max"); }
#line 3388 "source/parser.c"
    break;

  case 100:
#line 1204 "source/parser.y"
          { (yyval.symbol) = strdup("ceild"); }
#line 3394 "source/parser.c"
    break;

  case 101:
#line 1205 "source/parser.y"
           { (yyval.symbol) = strdup("floord"); }
#line 3400 "source/parser.c"
    break;

  case 102:
#line 1215 "source/parser.y"
    {
      int nb_columns;
      osl_relation_p id;
      osl_relation_list_p list;
      clan_symbol_p symbol;

      CLAN_debug("rule primary_expression.1: id_or_clan_keyword");
      symbol = clan_symbol_add(&parser_symbol, (yyvsp[0].symbol), CLAN_UNDEFINED);
      nb_columns = CLAN_MAX_DEPTH + CLAN_MAX_LOCAL_DIMS +
	                 CLAN_MAX_PARAMETERS + 2;
      id = osl_relation_pmalloc(parser_options->precision, 0, nb_columns);
      osl_relation_set_attributes(id, 0, parser_loop_depth, 0,
                                  CLAN_MAX_PARAMETERS);
      clan_relation_tag_array(id, symbol->key);
      list = osl_relation_list_malloc();
      list->elt = id;

      // add the id to the extbody
      if (parser_options->extbody) {
        if (parser_access_start != -1) {
          osl_extbody_add(parser_access_extbody,
                          parser_access_start,
                          parser_access_length);
        }

        int len = strlen(parser_record);
        parser_access_start = len - strlen((yyvsp[0].symbol));
        parser_access_length = len - parser_access_start;
      }

      free((yyvsp[0].symbol));
      (yyval.list) = list;
      CLAN_debug_call(osl_relation_list_dump(stderr, (yyval.list)));
    }
#line 3439 "source/parser.c"
    break;

  case 103:
#line 1250 "source/parser.y"
    { (yyval.list) = NULL; }
#line 3445 "source/parser.c"
    break;

  case 104:
#line 1252 "source/parser.y"
    { (yyval.list) = NULL; }
#line 3451 "source/parser.c"
    break;

  case 105:
#line 1254 "source/parser.y"
    { (yyval.list) = NULL; }
#line 3457 "source/parser.c"
    break;

  case 106:
#line 1256 "source/parser.y"
    { (yyval.list) = (yyvsp[-1].list); }
#line 3463 "source/parser.c"
    break;

  case 107:
#line 1262 "source/parser.y"
    { (yyval.list) = (yyvsp[0].list); }
#line 3469 "source/parser.c"
    break;

  case 108:
#line 1264 "source/parser.y"
    {
      if (parser_options->extbody)
        parser_access_length = strlen(parser_record) - parser_access_start;

      CLAN_debug("rule postfix_expression.2: postfix_expression [ <affex> ]");
      if (!clan_symbol_update_type(parser_symbol, (yyvsp[-3].list), CLAN_TYPE_ARRAY))
        YYABORT;
      clan_relation_new_output_vector((yyvsp[-3].list)->elt, (yyvsp[-1].affex));
      osl_vector_free((yyvsp[-1].affex));
      (yyval.list) = (yyvsp[-3].list);
      CLAN_debug_call(osl_relation_list_dump(stderr, (yyval.list)));
    }
#line 3486 "source/parser.c"
    break;

  case 109:
#line 1277 "source/parser.y"
    { 
      // don't save access name of a function
      if (parser_options->extbody) {
        parser_access_extbody->nb_access -= osl_relation_list_count((yyvsp[-2].list)) - 1;
        parser_access_start = -1;
      }

      if (!clan_symbol_update_type(parser_symbol, (yyvsp[-2].list), CLAN_TYPE_FUNCTION))
        YYABORT;
      osl_relation_list_free((yyvsp[-2].list));
      (yyval.list) = NULL;
    }
#line 3503 "source/parser.c"
    break;

  case 110:
#line 1290 "source/parser.y"
    {
      // don't save access name of a function
      if (parser_options->extbody) {
        parser_access_extbody->nb_access -= osl_relation_list_count((yyvsp[-1].list)) - 1;
        parser_access_start = -1;
      }
    }
#line 3515 "source/parser.c"
    break;

  case 111:
#line 1298 "source/parser.y"
    {
      if (!clan_symbol_update_type(parser_symbol, (yyvsp[-4].list), CLAN_TYPE_FUNCTION))
        YYABORT;
      osl_relation_list_free((yyvsp[-4].list));
      (yyval.list) = (yyvsp[-1].list);
    }
#line 3526 "source/parser.c"
    break;

  case 112:
#line 1305 "source/parser.y"
    {
      if (parser_options->extbody)
        parser_access_length = strlen(parser_record) - parser_access_start;

      clan_symbol_p symbol;

      CLAN_debug("rule postfix_expression.4: postfix_expression . "
                 "id_or_clan_keyword");
      if (!clan_symbol_update_type(parser_symbol, (yyvsp[-2].list), CLAN_TYPE_ARRAY))
        YYABORT;
      symbol = clan_symbol_add(&parser_symbol, (yyvsp[0].symbol), CLAN_TYPE_FIELD);
      clan_relation_new_output_scalar((yyvsp[-2].list)->elt, symbol->key);
      free((yyvsp[0].symbol));
      (yyval.list) = (yyvsp[-2].list);
      CLAN_debug_call(osl_relation_list_dump(stderr, (yyval.list)));
    }
#line 3547 "source/parser.c"
    break;

  case 113:
#line 1322 "source/parser.y"
    {
      if (parser_options->extbody)
        parser_access_length = strlen(parser_record) - parser_access_start;

      clan_symbol_p symbol;

      CLAN_debug("rule postfix_expression.5: postfix_expression -> "
                 "id_or_clan_keyword");
      if (!clan_symbol_update_type(parser_symbol, (yyvsp[-2].list), CLAN_TYPE_ARRAY))
        YYABORT;
      symbol = clan_symbol_add(&parser_symbol, (yyvsp[0].symbol), CLAN_TYPE_FIELD);
      clan_relation_new_output_scalar((yyvsp[-2].list)->elt, symbol->key);
      free((yyvsp[0].symbol));
      (yyval.list) = (yyvsp[-2].list);
      CLAN_debug_call(osl_relation_list_dump(stderr, (yyval.list)));
    }
#line 3568 "source/parser.c"
    break;

  case 114:
#line 1339 "source/parser.y"
    { 
      osl_relation_list_p list;

      CLAN_debug("rule postfix_expression.6: postfix_expression -> "
	         "postfix_expression ++/--");
      if (!clan_symbol_update_type(parser_symbol, (yyvsp[-1].list), CLAN_TYPE_ARRAY))
        YYABORT;
      list = (yyvsp[-1].list);
      // The last reference in the list is also written.
      if (list != NULL) {
        while (list->next != NULL)
          list = list->next;
        list->next = osl_relation_list_node(list->elt);
        list->next->elt->type = OSL_TYPE_WRITE;
      }
      (yyval.list) = (yyvsp[-1].list);
      CLAN_debug_call(osl_relation_list_dump(stderr, (yyval.list)));

      // add an empty line in the extbody
      if (parser_options->extbody) {
        osl_extbody_add(parser_access_extbody, -1, -1);
      }
    }
#line 3596 "source/parser.c"
    break;

  case 115:
#line 1366 "source/parser.y"
    { (yyval.list) = (yyvsp[0].list); }
#line 3602 "source/parser.c"
    break;

  case 116:
#line 1368 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-2].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3611 "source/parser.c"
    break;

  case 117:
#line 1376 "source/parser.y"
    { (yyval.list) = (yyvsp[0].list); }
#line 3617 "source/parser.c"
    break;

  case 118:
#line 1378 "source/parser.y"
    {
      osl_relation_list_p list;

      CLAN_debug("rule unary_expression.2: unary_expression -> "
	         "++/-- unary_expression");
      if (!clan_symbol_update_type(parser_symbol, (yyvsp[0].list), CLAN_TYPE_ARRAY))
        YYABORT;
      list = (yyvsp[0].list);
      // The last reference in the list is also written.
      if (list != NULL) {
        while (list->next != NULL)
          list = list->next;
        list->next = osl_relation_list_node(list->elt);
        list->next->elt->type = OSL_TYPE_WRITE;
      }
      (yyval.list) = (yyvsp[0].list);
      CLAN_debug_call(osl_relation_list_dump(stderr, (yyval.list)));

      // add an empty line in the extbody
      if (parser_options->extbody) {
        osl_extbody_add(parser_access_extbody, -1, -1);
      }
    }
#line 3645 "source/parser.c"
    break;

  case 119:
#line 1402 "source/parser.y"
    { (yyval.list) = (yyvsp[0].list); }
#line 3651 "source/parser.c"
    break;

  case 120:
#line 1404 "source/parser.y"
    { (yyval.list) = (yyvsp[0].list); }
#line 3657 "source/parser.c"
    break;

  case 121:
#line 1406 "source/parser.y"
    { (yyval.list) = NULL; }
#line 3663 "source/parser.c"
    break;

  case 130:
#line 1425 "source/parser.y"
    { (yyval.list) = (yyvsp[0].list); }
#line 3669 "source/parser.c"
    break;

  case 131:
#line 1427 "source/parser.y"
    { (yyval.list) = (yyvsp[0].list); }
#line 3675 "source/parser.c"
    break;

  case 132:
#line 1432 "source/parser.y"
    { (yyval.list) = (yyvsp[0].list); }
#line 3681 "source/parser.c"
    break;

  case 133:
#line 1434 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-2].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3690 "source/parser.c"
    break;

  case 134:
#line 1439 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-2].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3699 "source/parser.c"
    break;

  case 135:
#line 1444 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-2].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3708 "source/parser.c"
    break;

  case 136:
#line 1452 "source/parser.y"
    { (yyval.list) = (yyvsp[0].list); }
#line 3714 "source/parser.c"
    break;

  case 137:
#line 1454 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-2].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3723 "source/parser.c"
    break;

  case 138:
#line 1459 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-2].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3732 "source/parser.c"
    break;

  case 139:
#line 1467 "source/parser.y"
    { (yyval.list) = (yyvsp[0].list); }
#line 3738 "source/parser.c"
    break;

  case 140:
#line 1469 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-2].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3747 "source/parser.c"
    break;

  case 141:
#line 1474 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-2].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3756 "source/parser.c"
    break;

  case 142:
#line 1482 "source/parser.y"
    { (yyval.list) = (yyvsp[0].list); }
#line 3762 "source/parser.c"
    break;

  case 143:
#line 1484 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-2].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3771 "source/parser.c"
    break;

  case 144:
#line 1489 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-2].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3780 "source/parser.c"
    break;

  case 145:
#line 1494 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-2].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3789 "source/parser.c"
    break;

  case 146:
#line 1499 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-2].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3798 "source/parser.c"
    break;

  case 147:
#line 1507 "source/parser.y"
    { (yyval.list) = (yyvsp[0].list); }
#line 3804 "source/parser.c"
    break;

  case 148:
#line 1509 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-2].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3813 "source/parser.c"
    break;

  case 149:
#line 1514 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-2].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3822 "source/parser.c"
    break;

  case 150:
#line 1522 "source/parser.y"
    { (yyval.list) = (yyvsp[0].list); }
#line 3828 "source/parser.c"
    break;

  case 151:
#line 1524 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-2].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3837 "source/parser.c"
    break;

  case 152:
#line 1532 "source/parser.y"
    { (yyval.list) = (yyvsp[0].list); }
#line 3843 "source/parser.c"
    break;

  case 153:
#line 1534 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-2].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3852 "source/parser.c"
    break;

  case 154:
#line 1542 "source/parser.y"
    { (yyval.list) = (yyvsp[0].list); }
#line 3858 "source/parser.c"
    break;

  case 155:
#line 1544 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-2].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3867 "source/parser.c"
    break;

  case 156:
#line 1552 "source/parser.y"
    { (yyval.list) = (yyvsp[0].list); }
#line 3873 "source/parser.c"
    break;

  case 157:
#line 1554 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-2].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3882 "source/parser.c"
    break;

  case 158:
#line 1562 "source/parser.y"
    { (yyval.list) = (yyvsp[0].list); }
#line 3888 "source/parser.c"
    break;

  case 159:
#line 1564 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-2].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3897 "source/parser.c"
    break;

  case 160:
#line 1572 "source/parser.y"
    { (yyval.list) = (yyvsp[0].list); }
#line 3903 "source/parser.c"
    break;

  case 161:
#line 1574 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-4].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[-2].list));
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3913 "source/parser.c"
    break;

  case 162:
#line 1583 "source/parser.y"
    {
      CLAN_debug("rule assignment_expression.1: conditional_expression;");
      (yyval.list) = (yyvsp[0].list);
      clan_relation_list_define_type((yyval.list), OSL_TYPE_READ);
      CLAN_debug_call(osl_relation_list_dump(stderr, (yyval.list)));
    }
#line 3924 "source/parser.c"
    break;

  case 163:
#line 1590 "source/parser.y"
    {
      osl_relation_list_p list;

      CLAN_debug("rule assignment_expression.2: unary_expression "
	         "assignment_operator assignment_expression;");
      if (!clan_symbol_update_type(parser_symbol, (yyvsp[-2].list), CLAN_TYPE_ARRAY))
        YYABORT;
      (yyval.list) = (yyvsp[-2].list);
      // Accesses of $1 are READ except the last one which is a WRITE or both.
      clan_relation_list_define_type((yyval.list), OSL_TYPE_READ);
      list = (yyval.list);
      while (list->next != NULL)
        list = list->next;
      if ((yyvsp[-1].value) == CLAN_TYPE_RDWR) {
        list->next = osl_relation_list_node(list->elt);
        list = list->next;

        // add an empty line in the extbody
        if (parser_options->extbody) {
          osl_extbody_add(parser_access_extbody, -1, -1);
        }
      }
      osl_relation_set_type(list->elt, OSL_TYPE_WRITE);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
      CLAN_debug_call(osl_relation_list_dump(stderr, (yyval.list)));
    }
#line 3955 "source/parser.c"
    break;

  case 164:
#line 1620 "source/parser.y"
    { (yyval.value) = CLAN_TYPE_WRITE; }
#line 3961 "source/parser.c"
    break;

  case 165:
#line 1622 "source/parser.y"
    { (yyval.value) = CLAN_TYPE_RDWR; }
#line 3967 "source/parser.c"
    break;

  case 176:
#line 1640 "source/parser.y"
    { (yyval.list) = (yyvsp[0].list); }
#line 3973 "source/parser.c"
    break;

  case 177:
#line 1642 "source/parser.y"
    {
      (yyval.list) = (yyvsp[-2].list);
      osl_relation_list_add(&(yyval.list), (yyvsp[0].list));
    }
#line 3982 "source/parser.c"
    break;

  case 178:
#line 1650 "source/parser.y"
    {
      CLAN_debug("rule expression_statement.1: ;");
      (yyval.stmt) = NULL;
      CLAN_debug_call(osl_statement_dump(stderr, (yyval.stmt)));
    }
#line 3992 "source/parser.c"
    break;

  case 179:
#line 1656 "source/parser.y"
    {
      if (parser_options->extbody) {
        parser_access_start = -1;
        parser_access_extbody = osl_extbody_malloc();
      }

      CLAN_strdup(parser_record, scanner_latest_text);
      parser_recording = CLAN_TRUE;
    }
#line 4006 "source/parser.c"
    break;

  case 180:
#line 1666 "source/parser.y"
    {
      osl_statement_p statement;
      osl_body_p body;
      osl_generic_p gen;
      
      CLAN_debug("rule expression_statement.2: expression ;");
      statement = osl_statement_malloc();

      // - 1. Domain
      if (clan_relation_list_nb_elements(parser_stack->constraints) != 1) {
	yyerror("missing label on a statement inside an xfor loop");
        YYABORT;
      }
      statement->domain = osl_relation_clone(parser_stack->constraints->elt);
      osl_relation_set_type(statement->domain, OSL_TYPE_DOMAIN);
      osl_relation_set_attributes(statement->domain, parser_loop_depth, 0,
	                          clan_parser_nb_ld(), CLAN_MAX_PARAMETERS);

      // - 2. Scattering
      statement->scattering = clan_relation_scattering(parser_scattering,
          parser_loop_depth, parser_options->precision);

      // - 3. Array accesses
      statement->access = (yyvsp[-1].list);

      // - 4. Body.
      body = osl_body_malloc();
      body->iterators = clan_symbol_array_to_strings(parser_iterators,
	  parser_loop_depth, parser_xfor_depths, parser_xfor_labels);
      body->expression = osl_strings_encapsulate(parser_record);
      gen = osl_generic_shell(body, osl_body_interface());
      osl_generic_add(&statement->extension, gen);

      if (parser_options->extbody) {
        // Extended body

        // add the last access
        if (parser_access_start != -1) {
          osl_extbody_add(parser_access_extbody,
                          parser_access_start,
                          parser_access_length);
        }

        parser_access_extbody->body = osl_body_clone(body);
        gen = osl_generic_shell(parser_access_extbody, osl_extbody_interface());
        osl_generic_add(&statement->extension, gen);
      }

      parser_recording = CLAN_FALSE;
      parser_record = NULL;
      
      parser_scattering[2*parser_loop_depth]++;

      (yyval.stmt) = statement;
      CLAN_debug_call(osl_statement_dump(stderr, (yyval.stmt)));
    }
#line 4067 "source/parser.c"
    break;

  case 205:
#line 1767 "source/parser.y"
                                                                       { free((yyvsp[-3].symbol)); }
#line 4073 "source/parser.c"
    break;

  case 207:
#line 1769 "source/parser.y"
                                       { free((yyvsp[0].symbol)); }
#line 4079 "source/parser.c"
    break;

  case 223:
#line 1806 "source/parser.y"
                                                    { free((yyvsp[-3].symbol)); }
#line 4085 "source/parser.c"
    break;

  case 224:
#line 1807 "source/parser.y"
                            { free((yyvsp[0].symbol)); }
#line 4091 "source/parser.c"
    break;

  case 227:
#line 1816 "source/parser.y"
                       { free((yyvsp[0].symbol)); }
#line 4097 "source/parser.c"
    break;

  case 228:
#line 1817 "source/parser.y"
                                               { free((yyvsp[-2].symbol)); }
#line 4103 "source/parser.c"
    break;

  case 233:
#line 1831 "source/parser.y"
                       { free((yyvsp[0].symbol)); }
#line 4109 "source/parser.c"
    break;

  case 253:
#line 1870 "source/parser.y"
                       { free((yyvsp[0].symbol)); }
#line 4115 "source/parser.c"
    break;

  case 254:
#line 1871 "source/parser.y"
                                           { free((yyvsp[0].symbol)); }
#line 4121 "source/parser.c"
    break;


#line 4125 "source/parser.c"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = YY_CAST (char *, YYSTACK_ALLOC (YY_CAST (YYSIZE_T, yymsg_alloc)));
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYTERROR;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;


#if !defined yyoverflow || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif


/*-----------------------------------------------------.
| yyreturn -- parsing is finished, return the result.  |
`-----------------------------------------------------*/
yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  yystos[+*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  return yyresult;
}
#line 1897 "source/parser.y"



void yyerror(char *s) {
  int i, line = 1;
  char c = 'C';
  FILE* file;
 
  CLAN_debug("parse error notified");

  if (!parser_options->autoscop) {
    fprintf(stderr, "[Clan] Error: %s at line %d, column %d.\n", s,
        scanner_line, scanner_column - 1);

    // Print a message to show where is the problem.
    if ((parser_options != NULL) && (parser_options->name != NULL)) {
      file = fopen(parser_options->name, "r");
      if (file != NULL) {
        // Go to the right line.
        while (line != scanner_line) {
          c = fgetc(file);
          if (c == '\n')
            line++;
        }

        // Print the line.
        while (c != EOF) {
          c = fgetc(file);
          fprintf(stderr, "%c", c);
          if (c == '\n')
            break;
        }

        // Print the situation line.
        for (i = 0; i < scanner_column - 1; i++) {
          if (i < scanner_column - 5)
            fprintf(stderr, " ");
          else if (i < scanner_column - 2)
            fprintf(stderr, "~");
          else
            fprintf(stderr, "^\n");
        }
        fclose(file);
      } else {
        CLAN_warning("cannot open input file");
      }
    }
  }
  parser_error = CLAN_TRUE;
}


/**
 * clan_parser_state_print function:
 * this function "pretty" prints the parser state to a file.
 */
void clan_parser_state_print(FILE* file) {
  int i;

  fprintf(file, "+-- clan parser state\n");
  fprintf(file, "|\t|\n");

  // SCoP.
  fprintf(file, "|\tparser_scop [SCoP in construction]\n");
  fprintf(file, "|\t|\t|\n");
  osl_scop_idump(file, parser_scop, 2);
  fprintf(file, "|\t|\n");

  // Symbol table.
  fprintf(file, "|\tparser_symbol [Symbol table]\n");
  fprintf(file, "|\t|\t|\n");
  clan_symbol_print_structure(file, parser_symbol, 2);
  fprintf(file, "|\t|\n");

  // Recording boolean.
  fprintf(file, "|\tparser_recording [Boolean: do we record or not?] = %d\n",
          parser_recording);
  fprintf(file, "|\t|\n");

  // Recorded body.
  fprintf(file, "|\tparser_record [Statement body] = ");
  if (parser_record == NULL)
    fprintf(file, "(NULL)\n");
  else
    fprintf(file, "%s\n", parser_record);
  fprintf(file, "|\t|\n");

  // Loop depth.
  fprintf(file, "|\tparser_loop_depth [Current loop depth] = %d\n",
          parser_loop_depth);
  fprintf(file, "|\t|\n");

  // If depth.
  fprintf(file, "|\tparser_if_depth [Current if depth] = %d\n",
          parser_if_depth);
  fprintf(file, "|\t|\n");

  // Scattering.
  fprintf(file, "|\tparser_scattering [Current statement scattering]\n");
  fprintf(file, "|\t|\t|\n");
  fprintf(file, "|\t|\t+-- ");
  for (i = 0; i < 2 * parser_loop_depth + 1; i++)
    printf("%d ", parser_scattering[i]);
  fprintf(file, "\n");
  fprintf(file, "|\t|\t|\n");
  fprintf(file, "|\t|\n");

  // Iterators.
  fprintf(file, "|\tparser_iterators [Current iterator list]\n");
  fprintf(file, "|\t|\t|\n");
  if (parser_loop_depth > 0) {
    for (i = 0; i < parser_loop_depth; i++) {
      fprintf(file, "|\t|\tparser_iterators[%d]\n", i);
      fprintf(file, "|\t|\t|\t|\n");
      clan_symbol_print_structure(file, parser_iterators[i], 3);
      if (i == parser_loop_depth - 1)
	fprintf(file, "|\t|\t|\n");
    }
  } else {
    fprintf(file, "|\t|\t+-- (none)\n");
    fprintf(file, "|\t|\t|\n");
  }
  fprintf(file, "|\t|\n");

  // Iteration domain stack.
  fprintf(file, "|\tparser_domain [Iteration domain stack]\n");
  fprintf(file, "|\t|\t|\n");
  clan_domain_idump(file, parser_stack, 2);
  fprintf(file, "|\t|\n");

  // Number of local dimensions per depth.
  fprintf(file, "|\tparser_nb_local_dims [Nb of local dims per depth]\n");
  fprintf(file, "|\t|\t|\n");
  fprintf(file, "|\t|\t+-- ");
  if (parser_loop_depth > 0) {
    for (i = 0; i < parser_loop_depth; i++)
      printf("%d ", parser_nb_local_dims[i]);
      fprintf(file, "\n");
    } else {
    fprintf(file, "(none)\n");
  }
  fprintf(file, "|\t|\t|\n");
  fprintf(file, "|\t|\n");

  // Number of parameters.
  fprintf(file, "|\tparser_nb_parameters [Nb of parameter symbols] = %d\n",
          parser_nb_parameters);
  fprintf(file, "|\t|\n");

  // Boolean valid else per if depth.
  fprintf(file, "|\tparser_valid_else [Boolean: OK for else per depth]\n");
  fprintf(file, "|\t|\t|\n");
  fprintf(file, "|\t|\t+-- ");
  if (parser_if_depth > 0) {
    for (i = 0; i < parser_if_depth; i++)
      printf("%d ", parser_valid_else[i]);
    fprintf(file, "\n");
  } else {
    fprintf(file, "(none)\n");
  }
  fprintf(file, "|\t|\t|\n");
  fprintf(file, "|\t|\n");

  // Indentation.
  fprintf(file, "|\tparser_indent [SCoP indentation] = %d\n", parser_indent);
  fprintf(file, "|\t|\n");

  // Parse error boolean.
  fprintf(file, "|\tparser_error [Boolean: parse error] = %d\n", parser_error);
  fprintf(file, "|\t|\n");

  // xfor number of nests, depths and labels.
  fprintf(file, "|\txfor management [nb of nests, depths and labels]\n");
  fprintf(file, "|\t|\t|\n");
  if (parser_xfor_nb_nests > 0) {
    fprintf(file, "|\t|\t|  nest | depth | label\n");
    for (i = 0; i < parser_xfor_nb_nests; i++) {
      printf("|\t|\t|   [%d] |     %d |     %d\n",
	     i, parser_xfor_depths[i], parser_xfor_labels[i]);
    }
  } else {
    fprintf(file, "|\t|\t|  (no xfor loop)\n");
  }
  fprintf(file, "|\t|\t|\n");
  fprintf(file, "|\t|\n");
  
  // loop sanity sentinels
  fprintf(file, "|\tloop sanity sentinels [booleans min/max/floord/ceild]\n");
  fprintf(file, "|\t|\t|\n");
  if (parser_xfor_index > 0) {
    fprintf(file, "|\t|\t|  index | min | max | floord | ceild\n");
    for (i = 0; i < parser_xfor_index; i++) {
      printf("|\t|\t|  [%d] |   %d |   %d |      %d |     %d\n",
	     i, parser_min[i], parser_max[i],
	     parser_floord[i], parser_ceild[i]);
    }
  } else {
    fprintf(file, "|\t|\t|  (no (x)for loop indices)\n");
  }
  fprintf(file, "|\t|\t|\n");
  fprintf(file, "|\t|\n");
  
  fprintf(file, "|\n");
}


void clan_parser_add_ld() {
  parser_nb_local_dims[parser_loop_depth + parser_if_depth]++;

  if (CLAN_DEBUG) {
    int i;
    CLAN_debug("parser_nb_local_dims updated");
    for (i = 0; i <= parser_loop_depth + parser_if_depth; i++)
      fprintf(stderr, "%d:%d ", i, parser_nb_local_dims[i]);
    fprintf(stderr, "\n");
  }
  
  if (clan_parser_nb_ld() > CLAN_MAX_LOCAL_DIMS)
    CLAN_error("CLAN_MAX_LOCAL_DIMS reached, recompile with a higher value");
}


int clan_parser_nb_ld() {
  int i, nb_ld = 0;

  for (i = 0; i <= parser_loop_depth + parser_if_depth; i++)
    nb_ld += parser_nb_local_dims[i]; 
  return nb_ld;
}


void clan_parser_increment_loop_depth() {
  parser_loop_depth++;
  if ((parser_loop_depth + parser_if_depth) > CLAN_MAX_DEPTH)
    CLAN_error("CLAN_MAX_DEPTH reached, recompile with a higher value");
}


int clan_parser_is_loop_sane(osl_relation_list_p initialization,
                             osl_relation_list_p condition, int* stride) {
  int i, step;

  // Check there is the same number of elements in all for parts.
  if ((clan_relation_list_nb_elements(initialization) != parser_xfor_index) ||
      (clan_relation_list_nb_elements(condition) != parser_xfor_index)) {
    yyerror("not the same number of elements in all loop parts");
    return 0;
  }

  // Check that all bounds and strides are consistent.
  for (i = 0; i < parser_xfor_index; i++) {
    step = stride[i];
    if ((step == 0) ||
	((step > 0) && parser_min[i])    ||
	((step > 0) && parser_floord[i]) ||
	((step < 0) && parser_max[i])    ||
	((step < 0) && parser_ceild[i])) {
      osl_relation_list_free(initialization);
      osl_relation_list_free(condition);
      free(stride);
      if (step == 0)
	yyerror("unsupported zero loop stride");
      else if (step > 0)
	yyerror("illegal min or floord in forward loop initialization");
      else
	yyerror("illegal max or ceild in backward loop initialization");
      return 0;
    }
    parser_ceild[i]  = 0;
    parser_floord[i] = 0;
    parser_min[i]    = 0;
    parser_max[i]    = 0;
  }
  return 1;
}


/**
 * clan_parser_state_malloc function:
 * this function achieves the memory allocation for the "parser state".
 * \param[in] precision Precision of the integer elements.
 */
void clan_parser_state_malloc(int precision) {
  int nb_columns, depth;

  nb_columns        = CLAN_MAX_DEPTH + CLAN_MAX_LOCAL_DIMS +
                      CLAN_MAX_PARAMETERS + 2;
  depth             = CLAN_MAX_DEPTH;
  parser_stack      = clan_domain_malloc();
  parser_stack->constraints = osl_relation_list_malloc();
  parser_stack->constraints->elt = osl_relation_pmalloc(precision,
      0, nb_columns);
  CLAN_malloc(parser_nb_local_dims, int*, depth * sizeof(int));
  CLAN_malloc(parser_valid_else, int*, depth * sizeof(int));
  CLAN_malloc(parser_scattering, int*, (2 * depth + 1) * sizeof(int));
  CLAN_malloc(parser_iterators, clan_symbol_p*, depth * sizeof(clan_symbol_p));
  CLAN_malloc(parser_ceild,  int*, CLAN_MAX_XFOR_INDICES * sizeof(int));
  CLAN_malloc(parser_floord, int*, CLAN_MAX_XFOR_INDICES * sizeof(int));
  CLAN_malloc(parser_min,    int*, CLAN_MAX_XFOR_INDICES * sizeof(int));
  CLAN_malloc(parser_max,    int*, CLAN_MAX_XFOR_INDICES * sizeof(int));
  CLAN_malloc(parser_xfor_depths, int*, CLAN_MAX_DEPTH * sizeof(int));
  CLAN_malloc(parser_xfor_labels, int*, CLAN_MAX_DEPTH * sizeof(int));
}


/**
 * clan_parser_state_free function:
 * this function frees the memory allocated for the "parser state", with the
 * exception of the parser_scop.
 */
void clan_parser_state_free() {
  clan_symbol_free(parser_symbol);
  free(parser_scattering);
  free(parser_iterators);
  free(parser_nb_local_dims);
  free(parser_valid_else);
  free(parser_ceild);
  free(parser_floord);
  free(parser_min);
  free(parser_max);
  free(parser_xfor_depths);
  free(parser_xfor_labels);
  clan_domain_drop(&parser_stack);
}


/**
 * clan_parser_state_initialize function:
 * this function achieves the initialization of the "parser state", with
 * the exception of parser_scop.
 */
void clan_parser_state_initialize(clan_options_p options) {
  int i;

  parser_symbol        = NULL;
  parser_loop_depth    = 0;
  parser_options       = options;
  parser_recording     = CLAN_FALSE;
  parser_record        = NULL;
  parser_if_depth      = 0;
  parser_xfor_nb_nests = 0;
  parser_xfor_index    = 0;
  parser_indent        = CLAN_UNDEFINED;
  parser_error         = CLAN_FALSE;
  parser_autoscop      = CLAN_FALSE;
  parser_line_start    = 1;
  parser_line_end      = 1;
  parser_column_start  = 1;
  parser_column_end    = 1;
  parser_nb_parameters = 0;

  for (i = 0; i < CLAN_MAX_XFOR_INDICES; i++) {
    parser_ceild[i]  = 0;
    parser_floord[i] = 0;
    parser_min[i]    = 0;
    parser_max[i]    = 0;
  }

  for (i = 0; i < CLAN_MAX_DEPTH; i++) {
    parser_nb_local_dims[i] = 0;
    parser_valid_else[i] = 0;
    parser_iterators[i] = NULL;
    parser_xfor_depths[i] = 0;
    parser_xfor_labels[i] = CLAN_UNDEFINED;
  }

  for (i = 0; i < 2 * CLAN_MAX_DEPTH + 1; i++)
    parser_scattering[i] = 0;
}


/**
 * clan_parser_reinitialize function:
 * this function frees the temporary dynamic variables of the parser and
 * reset the variables to default values. It is meant to be used for a
 * clean restart after a parse error.
 */
void clan_parser_reinitialize() {
  int i;
  
  free(parser_record);
  clan_symbol_free(parser_symbol);
  for (i = 0; i < parser_loop_depth; i++)
    clan_symbol_free(parser_iterators[i]);
  while (parser_stack->next != NULL)
    clan_domain_drop(&parser_stack);
  osl_scop_free(parser_scop);
  clan_parser_state_initialize(parser_options);
}


/**
 * clan_parser_autoscop function:
 * this functions performs the automatic extraction of SCoPs from the input
 * file. It leaves the SCoP pragmas already set by the user intact (note that
 * as a consequence, user-SCoPs cannot be inserted to a larger SCoP).
 * It writes a file (named by the CLAN_AUTOPRAGMA_FILE macro) with the input
 * code where new SCoP pragmas have been inserted. If the option -autoscop
 * is set, it puts the list of SCoPs (including automatically discovered
 * SCoPs and user-SCoPs) in parser_scop.
 */
void clan_parser_autoscop() {
  int new_scop, nb_scops = 0;
  int line, column, restart_line, restart_column;
  long position;
  char c;
  int coordinates[5][CLAN_MAX_SCOPS]; // 0, 1: line start, end
                                      // 2, 3: column start, end
				      // 4: autoscop or not
 
  while (1) {
    // For the automatic extraction, we parse everything except user-SCoPs.
    if (!scanner_pragma)
      scanner_parsing = CLAN_TRUE;
    yyparse();

    new_scop = (parser_line_end != 1) || (parser_column_end != 1);
    restart_line = (new_scop) ? parser_line_end : scanner_line;
    restart_column = (new_scop) ? parser_column_end : scanner_column;
    if (CLAN_DEBUG) {
      if (new_scop)
	fprintf(stderr, "[Clan] Debug: new autoscop, ");
      else
	fprintf(stderr, "[Clan] Debug: no autoscop, ");
      fprintf(stderr, "restart at line %d, column %d\n",
	      restart_line, restart_column);
    }
 
    if (parser_error || new_scop) {
      if (new_scop) {
        // If a new SCoP has been found, store its coordinates.
        if (nb_scops == CLAN_MAX_SCOPS)
          CLAN_error("too many SCoPs! Change CLAN_MAX_SCOPS and recompile.");
        coordinates[0][nb_scops] = parser_line_start;
        coordinates[1][nb_scops] = parser_line_end;
        coordinates[2][nb_scops] = parser_column_start;
        coordinates[3][nb_scops] = parser_column_end;
        coordinates[4][nb_scops] = CLAN_TRUE;
        if (CLAN_DEBUG) {
          fprintf(stderr, "[Clan] Debug: AutoSCoP [%d,%d -> %d,%d]\n",
                  coordinates[0][nb_scops], coordinates[2][nb_scops],
                  coordinates[1][nb_scops], coordinates[3][nb_scops] - 1);
        }
        // Let's go for the next SCoP.
        parser_autoscop = CLAN_FALSE;
        nb_scops++;
      } else if (scanner_scop_start != CLAN_UNDEFINED) {
        // If the start of a user-SCoP is detected, store its coordinate.
	coordinates[0][nb_scops] = scanner_scop_start;
      } else if (scanner_scop_end != CLAN_UNDEFINED) {
        // If the end of a user-SCoP is detected, store its coordinate.
	coordinates[1][nb_scops] = scanner_scop_end;
	coordinates[2][nb_scops] = 0;
	coordinates[3][nb_scops] = 0;
	coordinates[4][nb_scops] = CLAN_FALSE;
        if (CLAN_DEBUG) {
          fprintf(stderr, "[Clan] Debug: user-SCoP [%d,%d -> %d,%d]\n",
                  coordinates[0][nb_scops], coordinates[2][nb_scops],
                  coordinates[1][nb_scops], coordinates[3][nb_scops]);
        }
	nb_scops++;
      }

      // Restart after the SCoP or after the error.
      rewind(yyin);
      line = 1;
      column = 1;
      while ((line != restart_line) || (column != restart_column)) {
        c = fgetc(yyin);
        column++;
        if (c == '\n') {
          line++;
          column = 1;
        }
      }
    }

    // Reinitialize the scanner and the parser for a clean restart.
    clan_scanner_free();
    clan_scanner_reinitialize(scanner_pragma, restart_line, restart_column);
    clan_parser_reinitialize();
    yyrestart(yyin);

    // Check whether we reached the end of file or not.
    position = ftell(yyin);
    c = fgetc(yyin);
    if (fgetc(yyin) == EOF)
      break;
    else 
      fseek(yyin, position, SEEK_SET);
  }
 
  // Write the code with the inserted SCoP pragmas in CLAN_AUTOPRAGMA_FILE.
  rewind(yyin);
  clan_scop_print_autopragma(yyin, nb_scops, coordinates);

  // Use the temporary file for usual parsing.
  scanner_line = 1;
  scanner_column = 1;
  scanner_pragma = CLAN_FALSE;
  parser_options->autoscop = CLAN_FALSE;
  if ((yyin = fopen(CLAN_AUTOPRAGMA_FILE, "r")) == NULL)
    CLAN_error("cannot create the temporary file");
  yyparse();
  fclose(yyin);

  // Update the SCoP coordinates with those of the original file.
  clan_scop_update_coordinates(parser_scop, coordinates);
  parser_options->autoscop = CLAN_TRUE;
  
  if (remove(CLAN_AUTOPRAGMA_FILE))
    CLAN_warning("cannot delete temporary file");
}


/**
 * clan_parse function:
 * this function parses a file to extract a SCoP and returns, if successful,
 * a pointer to the osl_scop_t structure.
 * \param input   The file to parse (already open).
 * \param options Options for file parsing.
 */
osl_scop_p clan_parse(FILE* input, clan_options_p options) {
  osl_scop_p scop;
  yyin = input;

  clan_parser_state_malloc(options->precision);
  clan_parser_state_initialize(options);
  clan_scanner_initialize();
  yyrestart(yyin);  //restart scanning another file
  parser_scop = NULL;

  if (!options->autoscop)
    yyparse();
  else
    clan_parser_autoscop();

  CLAN_debug("parsing done");

  clan_scanner_free();
  
  if (!parser_error)
    scop = parser_scop;
  else
    scop = NULL;

  clan_parser_state_free();
  CLAN_debug("parser state successfully freed");

  return scop;
}
