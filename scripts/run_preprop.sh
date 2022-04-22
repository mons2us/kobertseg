for ws in 1 2 4 5
do
    python preprocess.py -mode generate_sepdata \
                         -window_size $ws \
                         -use_stair
done