#!/bin/bash

AWA2_path="../data/AWA2"

[ ! -d $AWA2_path ] && mkdir $AWA2_path

https://cvml.ist.ac.at/AwA2/AwA2-base.zip
declare -a SubFolders=("AwA2-base" "AwA2-features")
for folder in ${SubFolders[@]}; do
   [ ! -d $AWA2_path/$folder ] &&
       echo ------------------------- && echo Start $folder load &&
       curl -o $AWA2_path/$folder.zip https://cvml.ist.ac.at/AwA2/$folder.zip &&
       echo Start $folder unzip
       unzip $AWA2_path/$folder.zip -d $AWA2_path/ &&
       rm $AWA2_path/$folder.zip
       echo $folder successfully loaded
done
echo Done
