#!/bin/bash

if [ $# -ne 2 ]; then 
	echo "not enough arguments"
	echo "usage: $0 <in_dir> <out_dir>"
	exit 2
fi

in_dir=$1
out_dir=$2

for file in $(ls $in_dir)
do
	movie_name=$(cut -d "." -f 1 <<< "$file")
	movie_scene_path="$out_dir/$movie_name"

	mkdir -p $movie_scene_path
	scenedetect -i $in_dir/$file save-images -n 1 -o $movie_scene_path
done
