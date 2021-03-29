#!/bin/bash 
set -o nounset -o errexit
VERSION=$1
if  [ `git status --porcelain=v1 | grep -v '^??'|wc -l ` -eq 0 ] ; then echo 'Good'; else {
    echo "Uncommitted changes found";
    exit 1;
} ; fi 
echo "$VERSION" > version.txt
echo "committing"
git commit -m "New version $VERSION" -v version.txt
echo "tagging"
git tag $VERSION
echo "preparing the pypi package"
rm -rf dist/*
python setup.py sdist bdist_wheel
twine check dist/*
twine upload dist/*
