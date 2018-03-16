#!/usr/bin/env bash
cd ../src


DB_NAME=intsys
URL=mongodb://markomihajlovicfm:itisme1994@ds115124.mlab.com:15124/intsys

sacredboard -mu ${URL} ${DB_NAME}  # run server

max_features=100
max_df=0.50
min_df=0.01
lowercase=True
stop_words='english'
analyzer='word'
strip_accents='unicode'

use_idf=True
sublinear_tf=True
norm=null

data_provider='reuters'
