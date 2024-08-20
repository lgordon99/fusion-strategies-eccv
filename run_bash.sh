setting=early_fusion
for i in {1..50}
do
    sbatch -p gpu_requeue -t 40 -n 1 --mem 40G --gres gpu:nvidia_a100-sxm4-40gb:1 -o bash-outputs/$setting/$setting_$i.out -e bash-errors/$setting/$setting_$i.err run_cnn.sh $setting $i 0.001 False
done

setting=late_fusion
for i in {1..50}
do
    sbatch -p gpu_requeue -t 30 -n 1 --mem 20G --gres gpu:nvidia_a100-sxm4-40gb:1 -o bash-outputs/$setting/$setting_$i.out -e bash-errors/$setting/$setting_$i.err run_cnn.sh $setting $i 0.001 False
done

setting=mixture_of_experts
for i in {1..50}
do
    sbatch -p gpu_requeue -t 40 -n 1 --mem 20G --gres gpu:nvidia_a100-sxm4-40gb:1 -o bash-outputs/$setting/$setting_$i.out -e bash-errors/$setting/$setting_$i.err run_cnn.sh $setting $i 0.0001 False
done
