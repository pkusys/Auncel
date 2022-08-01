#!/bin/sh

./bound sift10M 5000 5000 100 0.1

./bound deep10M 5000 5000 100 0.1

./bound gist 500 500 100 0.1

./bound text 5000 5000 100 0.1

./bound sift10M 5000 5000 50 0.1

./bound sift10M 5000 5000 10 0.1

./bound sift10M 5000 5000 100 0.05

./bound sift10M 5000 5000 100 0.01