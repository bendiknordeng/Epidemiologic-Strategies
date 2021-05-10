#!/bin/sh
# This is a comment!

echo make sure folder is Epidemiologic-Vaccine-Strategies/ ...
if [ $PWD != "/Users/sandernordeide/Epidemiologic-Vaccine-Strategies" ]
then cd Epidemiologic-Vaccine-Strategies/
fi

echo Choosing python module ...   
module load Python/3.8.2-GCCcore-9.3.0

echo running main.py with python3 ...

python3 main.py
