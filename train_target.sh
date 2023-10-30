bash_file_name=$(basename $0)
file_name=${bash_file_name/.sh/}
python_file_name=${bash_file_name/.sh/.py}
GPU=3
VERSION=$file_name
date=$(date "+%Y-%m-%d")
output_dir=output/${date}_${VERSION}
if  [ ! -d  $output_dir ]; then
    mkdir $output_dir
fi
if  [ ! -d  $output_dir/checkpoints ]; then
    mkdir $output_dir/checkpoints
fi
if  [ -d ${output_dir}/$date.log ]; then
    rm -rf ${output_dir}/$date.log
fi
if  [ -d ${output_dir}/log ]; then
    rm -rf ${output_dir}/log
fi
cp *.* $output_dir

# high_thres=0.9
# low_thres=0.0
# weight=0.5
# dy_epoch=100
# nepoch=200
# sk=256
# tk=256
# sk_ratio=0.05
# tk_ratio=0.05
# iter_per_epoch=100
# dataset=office31
# data_dir=/data3/ywzhao/data/$dataset
# common_args="--high_thres $high_thres --low_thres $low_thres --gpu $GPU --output_dir $output_dir --date $date --weight $weight --dy_epoch $dy_epoch --dataset $dataset --nepoch $nepoch --sk $sk --tk $tk --sk_ratio $sk_ratio --tk_ratio $tk_ratio --iter_per_epoch $iter_per_epoch"
# echo $(date "+%Y-%m-%d %H:%M:%S")
# python $python_file_name --root_path $data_dir --src amazon --tar webcam $common_args
# echo $(date "+%Y-%m-%d %H:%M:%S")
# python $python_file_name --root_path $data_dir --src dslr --tar amazon $common_args
# echo $(date "+%Y-%m-%d %H:%M:%S")
# python $python_file_name --root_path $data_dir --src webcam --tar amazon $common_args
# echo $(date "+%Y-%m-%d %H:%M:%S")
# python $python_file_name --root_path $data_dir --src dslr --tar webcam $common_args
# echo $(date "+%Y-%m-%d %H:%M:%S")
# python $python_file_name --root_path $data_dir --src webcam --tar dslr $common_args
# echo $(date "+%Y-%m-%d %H:%M:%S")
# python $python_file_name --root_path $data_dir --src amazon --tar dslr $common_args

high_thres=0.9
low_thres=0.0
weight=0.5
dy_epoch=100
nepoch=200
sk=256
tk=256
sk_ratio=0.05
tk_ratio=0.05
iter_per_epoch=100
dataset=office-home
data_dir=/data3/ywzhao/data/$dataset
common_args="--high_thres $high_thres --low_thres $low_thres --gpu $GPU --output_dir $output_dir --date $date --weight $weight --dy_epoch $dy_epoch --dataset $dataset --nepoch $nepoch --sk $sk --tk $tk --sk_ratio $sk_ratio --tk_ratio $tk_ratio --iter_per_epoch $iter_per_epoch"
echo $(date "+%Y-%m-%d %H:%M:%S")
python $python_file_name --root_path $data_dir --src Art --tar Clipart $common_args
echo $(date "+%Y-%m-%d %H:%M:%S")
python $python_file_name --root_path $data_dir --src Art --tar Product $common_args
echo $(date "+%Y-%m-%d %H:%M:%S")
python $python_file_name --root_path $data_dir --src Art --tar Real_World $common_args
echo $(date "+%Y-%m-%d %H:%M:%S")
python $python_file_name --root_path $data_dir --src Clipart --tar Art $common_args
echo $(date "+%Y-%m-%d %H:%M:%S")
python $python_file_name --root_path $data_dir --src Clipart --tar Product $common_args
echo $(date "+%Y-%m-%d %H:%M:%S")
python $python_file_name --root_path $data_dir --src Clipart --tar Real_World $common_args
echo $(date "+%Y-%m-%d %H:%M:%S")
python $python_file_name --root_path $data_dir --src Product --tar Art $common_args
echo $(date "+%Y-%m-%d %H:%M:%S")
python $python_file_name --root_path $data_dir --src Product --tar Clipart $common_args
echo $(date "+%Y-%m-%d %H:%M:%S")
python $python_file_name --root_path $data_dir --src Product --tar Real_World $common_args
echo $(date "+%Y-%m-%d %H:%M:%S")
python $python_file_name --root_path $data_dir --src Real_World --tar Art $common_args
echo $(date "+%Y-%m-%d %H:%M:%S")
python $python_file_name --root_path $data_dir --src Real_World --tar Clipart $common_args
echo $(date "+%Y-%m-%d %H:%M:%S")
python $python_file_name --root_path $data_dir --src Real_World --tar Product $common_args

high_thres=0.9
low_thres=0.0
weight=0.5
dy_epoch=50
nepoch=100
sk=2048
tk=2048
iter_per_epoch=500
backbone=resnet101
dataset=visda17
data_dir=/data3/ywzhao/data/$dataset
common_args="--high_thres $high_thres --low_thres $low_thres --gpu $GPU --output_dir $output_dir --date $date --weight $weight --dy_epoch $dy_epoch --dataset $dataset --iter_per_epoch $iter_per_epoch --nepoch $nepoch --iter_based --sk $sk --tk $tk --backbone $backbone"
echo $(date "+%Y-%m-%d %H:%M:%S")
python $python_file_name --root_path $data_dir --src train --tar validation $common_args