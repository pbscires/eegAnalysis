#!/bin/bash

echo "directory to convert = $1"
edfDir=/Users/rsburugula/Documents/Etc/Pranav/YHS/ScienceResearch/Data/edf/chbmit13_24/$1
mitDir=/Users/rsburugula/Documents/Etc/Pranav/YHS/ScienceResearch/Data/mit/chbmit13_24/$1
converterExec=/Users/rsburugula/Documents/Etc/Pranav/YHS/ScienceResearch/code/wfdb-tools/wfdb-10.5.8-i386-Darwin/usr/bin/edf2mit
converterLib=/Users/rsburugula/Documents/Etc/Pranav/YHS/ScienceResearch/code/wfdb-tools/wfdb-10.5.8-i386-Darwin/usr/lib

cd $edfDir
for file in `ls *.edf`
do
   echo $file
   DYLD_LIBRARY_PATH=$converterLib $converterExec -i $file
done

mkdir -p $mitDir
mv *.dat $mitDir
mv *.hea $mitDir
