#!/bin/sh

# A Unix shell script for cross-validation subset generation scripts

# Description: this script is used to split the ratings data 
#              for k-fold cross-validation of rating predictions. 

# Note: By default produces a five fold splitting.

# Source:
#       http://files.grouplens.org/datasets/movielens/ml-10m-README.html

# Usage: chmod +x split_ratings.sh 
#        ./split_ratings.sh <data_dir> <data_file>
# -----------------------------------------------------------------------------


DATA_PATH=$1/$2 #./data/movielens_10m/ratings_implicit.txt
DATA_DIR=$1
RATINGS_COUNT=`wc -l $DATA_PATH | xargs | cut -d ' ' -f 1`
echo "ratings count: $RATINGS_COUNT"
SET_SIZE=`expr $RATINGS_COUNT / 5`
echo "test set size: $SET_SIZE"
TRAIN_SIZE=`expr $RATINGS_COUNT - $SET_SIZE`
echo "train set size: $TRAIN_SIZE"
REMAINDER=`expr $RATINGS_COUNT % 5`
echo "remainder: $REMAINDER"

# MIN_FOLD_NUM=`expr 0`
# MAX_FOLD_NUM=`expr 4`
# for i in 0 1 2 3 4 does not work!

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

   echo "Test_ratings_fold_$i created. `wc -l $DATA_DIR/Test_ratings_fold_$i | xargs | cut -d " " -f 1` lines."
   echo "Train_ratings_fold_$i created. `wc -l $DATA_DIR/Train_ratings_fold_$i | xargs | cut -d " " -f 1` lines."
done


