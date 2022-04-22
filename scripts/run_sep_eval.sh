declare -A ws_dict=(['NE01']=1 ['NE02']=2 ['NF03']=3 ['NE04']=4 ['NE05']=5 ['E05']=5)
# declare -A maxW_dict=(['week']=20 ['month']=8)
# declare -A name_dict=(['madgan']='madgan')

# target_name='A'
# criterion=${cri_dict[${target_name}]}
# max_window=${maxW_dict[$criterion]}

#thresh=0.85

for id in NF03
do
    for thresh in `seq 0.85 0.05 0.9`
    do
        ws=${ws_dict[${id}]}
        for compare in True
        do
            for num in `seq 10000 10000 180000`
            do
                echo $id window_size: $ws threshold: $thresh
                python main.py \
                        --mode test \
                        --test_mode sep \
                        --visible_gpus 1 \
                        --test_from models/index_${id}/model_w${ws}_fixed_step_$num.pt \
                        --data_type bfly \
                        --test_sep_num -1 \
                        --threshold $thresh \
                        --compare_window $compare
            done
        done
    done
done