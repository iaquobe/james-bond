#!/bin/bash 
files=("/home/iaquobe/Courses/Cultural-Analytics/james-bond/scenes/1964-goldfinger/1964-goldfinger-Scene-411-01.jpg")


for file in $files
do
	python src/main.py scenes analysis -d -s $file 
done

