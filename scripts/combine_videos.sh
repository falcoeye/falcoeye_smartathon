#!/usr/bin/env bash


DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
path=$1
ls -l $path
cd $1
echo "" > files.txt
for i in `ls`;do
echo "file $i" >> files.txt
done
cat files.txt
ffmpeg -f concat -i files.txt  -c copy $2.mp4
cd -
