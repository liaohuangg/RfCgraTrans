#! /bin/sh
srcdir=/root/workspace/RfCgraTrans/pluto/cloog-isl
PACKAGE=cloog
VERSION=0.20.0

if test -f $srcdir/.git/HEAD; then
    GIT_REPO="$srcdir/.git"
    GIT_HEAD_ID="$PACKAGE-$VERSION-`GIT_DIR=$GIT_REPO git rev-parse --short HEAD`"
elif test -f $srcdir/CLOOG_HEAD; then
    GIT_HEAD_ID=`cat $srcdir/CLOOG_HEAD`
else
    GIT_HEAD_ID="$PACKAGE-$VERSION-UNKNOWN"
fi

echo $GIT_HEAD_ID | sed -e 's/cloog-//'
