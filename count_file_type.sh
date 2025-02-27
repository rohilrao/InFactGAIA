#!/bin/bash
find . -type f | awk -F. 'NF>1 {print $NF} NF==1 {print "no_extension"}' | sort | uniq -c | sort -nr

