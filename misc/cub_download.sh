#!/bin/bash

CUB_path="../data/CUB-200"

[ ! -d $CUB_path ] && mkdir $CUB_path
declare -a SubFolders=("images" "lists" "annotations" "attributes")
for folder in ${SubFolders[@]}; do
   [ ! -d $CUB_path/$folder ] &&
       echo ------------------------- && echo Start $folder load &&
       curl -o $CUB_path/$folder.tgz http://www.vision.caltech.edu/visipedia-data/CUB-200/$folder.tgz &&
       echo Start $folder unzip
       tar -xzf $CUB_path/$folder.tgz -C $CUB_path/ &&
       rm $CUB_path/$folder.tgz
       echo $folder successfully loaded
done
rm $CUB_path/._*
rm $CUB_path/*/._*
rm $CUB_path/*/*/._*
echo Done
