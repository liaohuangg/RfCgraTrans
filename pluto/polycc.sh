#!/bin/bash
#
# Top-level script that runs all components of the end-to-end
# system
#
# Just run 'polycc <C code>' when the program section to
# be parallelized/optimized around special comments as described
# in the `README'
#
# Copyright (C) 2007-2008 Uday Bondhugula
#
# This file is available under the MIT license. Please see LICENSE in the
# top-level directory for details.
#
pluto=/root/workspace/RfCgraTrans/pluto/tool/pluto
inscop=/root/workspace/RfCgraTrans/pluto/inscop

# Some additional setup here to ensure that variables are visible outside of the
# run function.
SOURCEFILE=""
OUTFILE=""
dirname=""
PLUTOOUT=""

# check for command-line options
for arg in $*; do
    if [ $arg == "--parallel" ]; then
        PARALLEL=1
    elif [ $arg == "--parallelize" ]; then
        PARALLEL=1
    elif [ $arg == "--unroll" ]; then
        UNROLL=1
    elif [ $arg == "--debug" ]; then
        DEBUG=1
    elif [ $arg == "--moredebug" ]; then
        DEBUG=1
    elif [ $arg == "-i" ]; then
        INDENT=1
    elif [ $arg == "--indent" ]; then
        INDENT=1
    elif [ $arg == "--silent" ]; then
        SILENT=1
    fi
done

# some special processing for linearized accesses
#if [ "$SOURCEFILE" != "" ]; then
#grep __SPECIAL $SOURCEFILE > .nonlinearized
#grep __SPECIAL $SOURCEFILE | sed -e "s/.*__SPECIAL//" > .linearized
#fi

run()
{
$pluto $* || exit 1

SOURCEFILE=`cat .srcfilename`
OUTFILE=`cat .outfilename`

dirname=`dirname  $SOURCEFILE`
basename=`basename $SOURCEFILE`
prefix=`basename $SOURCEFILE .c`

CLOOGFILE=`basename $OUTFILE`.pluto.cloog
PLUTOOUT=$OUTFILE

# put the original skeleton around the transformed code
$inscop $SOURCEFILE $OUTFILE $OUTFILE

if [ "$INDENT" == 1 ] && [ -x /usr/bin/clang-format ]; then
    clang-format --style=LLVM -i $OUTFILE
fi
}

run "$*"
WORK=1
TEMPFILE=""
while [ $WORK -eq 1 ]
do
  if grep -q "#pragma scop" "$PLUTOOUT"
  then
    # Move the original file into a temporary location
    TEMPFILE="$SOURCEFILE""_temp"
    mv $SOURCEFILE $TEMPFILE

    # Move the file that still has scope in it into
    # place of the original source file, so $* will pick the
    # correct file
    mv $PLUTOOUT $SOURCEFILE

    # Run pluto again
    run "$*"

    # Move the original back in place
    mv $TEMPFILE $SOURCEFILE
  else
    # No more scops
    WORK=0
  fi
done

cleanup() {
  # An attempt to move the original file back in place
  # in the event of an exception.
  if [ -f "$TEMPFILE" ]
  then
    mv $TEMPFILE $SOURCEFILE
  fi
  if [ "$DEBUG" != 1 ];
  then
    rm -f .regtile .vectorize .pragmas .params .orcc .linearized .nonlinearized \
      $CLOOGFILE .srcfilename .outfilename .distmem pi.cloog sigma.cloog \
      *.sysloog .appendfilename
  fi
}

trap cleanup SIGINT exit
