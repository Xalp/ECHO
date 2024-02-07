# Define the array of dataset names
dataset_names=("multiarith" "gsm8k" "singleeq" "addsub" "aqua" "svamp" "commonsensqa" "strategyqa" "last_letters" "coin_flip") # Add more dataset names as needed
model_name="gpt-3.5-turbo-0301"

# Loop over dataset names
for dataset_name in "${dataset_names[@]}"; do

    # Loop over cluster numbers from 8 to 32
    for num_clusters in {8..32}; do

        python run_demo.py \
            --task $dataset_name \
            --pred_file log/${dataset_name}_${model_name}_zero_shot_cot.log \
            --demo_save_dir demos/${dataset_name}_${model_name}_${num_clusters} \
            --num_clusters ${num_clusters}

    done
done