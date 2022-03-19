#!/bin/bash

mkdir stereo_trainset/crestereo -p
cd stereo_trainset/crestereo


for dir in tree shapenet reflective hole
do
  mkdir $dir && cd $dir
  for i in $(seq 0 9)
  do
    echo $dir: $(expr $i + 1) / 10
    wget https://data.megengine.org.cn/research/crestereo/dataset/$dir/$i.tar
  done
  cd ..
done

for dir in tree shapenet reflective hole
do
  cd $dir
  for i in $(seq 0 9)
  do
    tar -xvf $i.tar
    rm $i.tar
  done
  cd ..
done





