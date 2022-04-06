#!/bin/bash
for i in $(seq -f "%02g" 0 10)
do
	python __main__.py -p ../ -s $i
done
