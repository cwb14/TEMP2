#!/bin/bash

if [ $# -lt 1 ];then
	echo -e "$0 output_prefix"
	echo -e "get the distance of 5' and 3' supporting reads without using the information of clipping"
	exit 1
fi

mv $1".insertion.bed" $1".insertion.bed.copy"
gawk 'BEGIN{FS=OFS="\t"} {if(ARGIND==1){l=$3-$2;p[$1":"$2]=l;p[$1":"$3]=l}else{if(p[$1":"$2]){l=p[$1":"$2]}else if(p[$1":"$3]){l=p[$1":"$3]}else{l=0};print $0,l}}' `dirname $1`/tmpTEMP2/`basename ${1}`.final.bed $1".insertion.bed.copy" > $1".insertion.bed" 

