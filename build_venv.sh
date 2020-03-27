#!/usr/bin/env bash

printf "The script will create a LOCAL PYTHON VIRTUAL ENVIRONMENT to run the package.
Continue? (Y/N)\n"
read -p "" -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

printf "\nStep 1/2: Virtual Environment building"
pip3 install virtualenv
rm -rf .venv/
virtualenv -p python3 .venv
source .venv/bin/activate

printf "\nStep 2/2: Python packages installation"
pip install -r requirements.txt

pip install -e .

rm -rf *.egg-info
printf "\nBuild is successfully complete!"
printf "\nYou can use launch the package with Python! For example of usage, see README."