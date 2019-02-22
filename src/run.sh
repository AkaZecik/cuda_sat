#!/bin/bash

set -e

make

for i in $(seq 1 100) ; do
	echo "Running ${i}"
	./a.out < "./input75_325/uf75-0${i}.cnf2" | grep -P '^satisfied$'
done
