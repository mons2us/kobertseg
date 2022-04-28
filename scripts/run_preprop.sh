for ws in 3
do
    python preprocess.py -mode generate_sepdata \
                         -window_size $ws \
                         -use_stair
done