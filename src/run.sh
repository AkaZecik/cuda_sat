#!/bin/bash

set -e

make

for i in $(seq 1 1000) ; do
	echo "Running ${i}"
	./a.out < "./input20_91/uf20-0${i}.cnf2" | grep -P '^satisfied$'
done
