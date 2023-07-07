#!/bin/bash

a=1
num=1

for f in * ; do
    mv -- "$f" "${a}"".jpeg"
    a=$(($a+$num))
    echo $a
done
