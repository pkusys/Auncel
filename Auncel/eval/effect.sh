#!/bin/sh

./effect_error sift10M 100 5000 5000

./effect_error deep10M 100 5000 5000

./effect_error gist 100 500 500

./effect_error text 100 5000 5000

./effect_time sift10M 100 5000 5000

./effect_time deep10M 100 5000 5000

./effect_time gist 100 500 500

./effect_time text 100 5000 5000