#!/bin/sh

# cross-validation subset generation scripts

# Description: this script is used to split the ratings data 
#              for five-fold cross-validation of rating predictions. 

# Source:
#       http://files.grouplens.org/datasets/movielens/ml-10m-README.html

# Usage: chmod +x split_ratings.sh 
#        ./split_ratings.sh <data_dir> <data_file>
# -----------------------------------------------------------------------------


DATA_PATH=$1/$2
DATA_DIR=$1
RATINGS_COUNT=`wc -l $DATA_PATH | xargs | cut -d ' ' -f 1`
echo "ratings count: $RATINGS_COUNT"
SET_SIZE=`expr $RATINGS_COUNT / 5`
echo "set size: $SET_SIZE"
REMAINDER=`expr $RATINGS_COUNT % 5`
echo "remainder: $REMAINDER"

for i in 1 2 3 4 5
  do
    head -`expr $i \* $SET_SIZE` $DATA_PATH | tail -$SET_SIZE > $DATA_DIR/Test_ratings_fold_$i

    # XXX: OSX users will see the message "head: illegal line count -- 0" here,
    #      but this is just a warning; the script still works as intended.
    head -`expr \( $i - 1 \) \* $SET_SIZE` $DATA_PATH > $DATA_DIR/Train_ratings_fold_$i
    tail -`expr \( 5 - $i \) \* $SET_SIZE` $DATA_PATH >> $DATA_DIR/Train_ratings_fold_$i

    if [ $i -eq 5 ]; then
       tail -$REMAINDER $DATA_PATH >> $DATA_DIR/Test_ratings_fold_5
    else
       tail -$REMAINDER $DATA_PATH >> $DATA_DIR/Train_ratings_fold_$i
    fi

   echo "Test_ratings_fold_$i created. `wc -l r$i.test | xargs | cut -d " " -f 1` lines."
   echo "Train_ratings_fold_$i created. `wc -l r$i.train | xargs | cut -d " " -f 1` lines."
done


