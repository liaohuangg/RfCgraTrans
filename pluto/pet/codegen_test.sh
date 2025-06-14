#!/bin/sh

EXEEXT=
srcdir=/root/workspace/RfCgraTrans/pluto/isl

for i in $srcdir/test_inputs/codegen/*.st \
		$srcdir/test_inputs/codegen/cloog/*.st; do
	echo $i;
	for opt in "" "--separate" "--atomic" \
		"--isl-no-ast-build-atomic-upper-bound"; do
		echo options: $opt
		(./pet_codegen$EXEEXT --tree $opt < $i > test.c &&
		 ./pet_check_code$EXEEXT --tree $i test.c) || exit
	done
done

for i in $srcdir/test_inputs/codegen/*.in \
		$srcdir/test_inputs/codegen/omega/*.in \
		$srcdir/test_inputs/codegen/pldi2012/*.in; do
	echo $i;
	for opt in "" "--separate" "--atomic" \
		"--isl-no-ast-build-atomic-upper-bound" "--read-options"; do
		echo options: $opt
		(./pet_codegen$EXEEXT $opt < $i > test.c &&
		 ./pet_check_code$EXEEXT $i test.c) || exit
	done
done

rm test.c
