#!/bin/sh

./bound sift10M 5000 5000 100 0.1 1

./bound deep10M 5000 5000 100 0.1 2

./bound gist 500 500 100 0.1 3

./bound text 5000 5000 100 0.1 4

./bound sift10M 5000 5000 50 0.1 5

./bound sift10M 5000 5000 10 0.1 6 

./bound sift10M 5000 5000 100 0.05 7

./bound sift10M 5000 5000 100 0.01 8