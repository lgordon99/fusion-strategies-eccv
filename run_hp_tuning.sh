declare -A ram
ram['early_fusion']=40
ram['late_fusion']=20
ram['mixture_of_experts']=20

for setting in 'early_fusion' 'late_fusion' 'mixture_of_experts'; do
    for trial in {1..10}; do
        for lr in 0.1 0.01 0.001 0.0001 0.00001; do
            sbatch -p gpu_requeue -t 60 -n 1 --mem ${ram[$setting]}G --gres gpu:nvidia_a100-sxm4-40gb:1 -o bash-outputs/$setting/lr_$lr/trial_$trial.out -e bash-errors/$setting/lr_$lr/trial_$trial.err run_cnn.sh $setting $trial $lr True
        done
    done
done
