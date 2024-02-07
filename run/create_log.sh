# Define the array of dataset names
dataset_names=("multiarith" "gsm8k" "singleeq" "addsub" "aqua" "svamp" "commonsensqa" "strategyqa" "last_letters" "coin_flip") # Add more dataset names as needed
model_name="gpt-3.5-turbo-0301"

# Loop over dataset names
for dataset_name in "${dataset_names[@]}"; do

    python zero_shot_cot.py >> log/${dataset_name}_${model_name}_zero_shot_cot.log \
        --dataset ${dataset_name} \
        --method zero_shot_cot \
        --model ${model_name} \
        --limit_dataset_size 0

done