#!/bin/sh
# getversion.h will be automatically generated from this
# Output of getversion.h goes into src/version.h

githead=/root/workspace/RfCgraTrans/pluto/.git/HEAD
version=0.12.0

if [ -f $githead ]; then
    echo `git describe --tags --always`
else
    echo $version
fi
