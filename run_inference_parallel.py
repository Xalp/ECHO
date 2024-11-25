import argparse
import json
import os
import numpy as np
from multiprocessing import Pool, cpu_count
from utils import *  # Ensure all your utility functions (e.g., `Decoder`, `fix_seed`) are imported

def process_data(data_args):
    """Process a single data sample."""
    i, data, args, demo, decoder = data_args
    output_line = {}
    x, y = data
    x = "Q: " + x[0] + "\n" + "A:"
    y = y[0].strip()

    output_line["question"] = x
    output_line["gold_ans"] = y

    if args.method == "zero_shot":
        x = x + " " + args.direct_answer_trigger_for_zeroshot
    elif args.method == "zero_shot_cot":
        x = x + " " + args.cot_trigger
    elif args.method in ["few_shot", "few_shot_cot", "auto_cot"]:
        x = demo + x + (" " + args.cot_trigger if "cot" in args.method else "")
    else:
        raise ValueError("method is not properly defined ...")

    # Generate the rationale and prediction
    max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
    z = decoder.decode(args, x, max_length)
    output_line["rationale"] = z

    if args.method == "zero_shot_cot":
        z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
        pred = decoder.decode(args, z2, args.max_length_direct)
    else:
        pred = z

    # Cleanse and store the prediction
    pred = answer_cleansing(args, pred)
    output_line["pred_ans"] = pred
    output_line["wrap_que"] = x

    correct = (np.array([pred]) == np.array([y])).sum().item()
    return i, output_line, correct

def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT Parallel Inference")

    parser.add_argument("--random_seed", type=int, default=1, help="Random seed")
    parser.add_argument("--dataset", type=str, default="multiarith", choices=[
        "aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", 
        "svamp", "singleeq", "coin_flip", "last_letters"], help="Dataset used for the experiment")
    parser.add_argument("--demo_path", type=str, default="demos/multiarith_manual", help="Path to pre-generated demos")
    parser.add_argument("--resume_id", type=int, default=0, help="Resume processing from a specific question ID")
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="Minibatch size (1 for GPT-3 API)")
    parser.add_argument("--max_num_worker", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--model", type=str, default="gpt3-xl", help="Model used for decoding")
    parser.add_argument("--method", type=str, default="auto_cot", choices=[
        "zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot"], help="Method type")
    parser.add_argument("--output_dir", type=str, default="experiment/multiarith", help="Output directory")
    parser.add_argument("--max_length_cot", type=int, default=1024, help="Max tokens for reasoning extraction")
    parser.add_argument("--max_length_direct", type=int, default=32, help="Max tokens for direct answer extraction")
    parser.add_argument("--limit_dataset_size", type=int, default=0, help="Limit dataset size for testing (0 = no limit)")
    parser.add_argument("--api_time_interval", type=float, default=0, help="Time interval between API calls")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for GPT-3")
    parser.add_argument("--log_dir", type=str, default="./log/", help="Log directory")
    
    args = parser.parse_args()

    # Define dataset paths and triggers
    dataset_settings = {
        "aqua": ("./dataset/AQuA/test.json", "\nTherefore, among A through E, the answer is"),
        "gsm8k": ("./dataset/grade-school-math/test.jsonl", "\nTherefore, the answer (arabic numerals) is"),
        "commonsensqa": ("./dataset/CommonsenseQA/dev_rand_split.jsonl", "\nTherefore, among A through E, the answer is"),
        "addsub": ("./dataset/AddSub/AddSub.json", "\nTherefore, the answer (arabic numerals) is"),
        "multiarith": ("./dataset/MultiArith/MultiArith.json", "\nTherefore, the answer (arabic numerals) is"),
        "strategyqa": ("./dataset/StrategyQA/task.json", "\nTherefore, the answer (Yes or No) is"),
        "svamp": ("./dataset/SVAMP/SVAMP.json", "\nTherefore, the answer (arabic numerals) is"),
        "singleeq": ("./dataset/SingleEq/questions.json", "\nTherefore, the answer (arabic numerals) is"),
        "coin_flip": ("./dataset/coin_flip/coin_flip.json", "\nTherefore, the answer (Yes or No) is"),
        "last_letters": ("./dataset/last_letters/last_letters.json", "\nTherefore, the answer is"),
    }

    if args.dataset in dataset_settings:
        args.dataset_path, args.direct_answer_trigger = dataset_settings[args.dataset]
    else:
        raise ValueError("Dataset not properly defined")

    # Configure triggers
    args.direct_answer_trigger_for_zeroshot = args.direct_answer_trigger.replace("\nTherefore, ", "").capitalize()
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."
    
    return args

def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')

    fix_seed(args.random_seed)
    print("OPENAI_API_KEY:")
    print(os.getenv("OPENAI_API_KEY")[0:5] + '**********')

    # Initialize decoder class (load model and tokenizer)
    decoder = Decoder()

    print("Setting up data loader ...")
    dataloader = setup_data_loader(args)
    print_now()

    # Create demonstration text for few-shot methods
    demo = create_demo_text(args, cot_flag="cot" in args.method) if "few_shot" in args.method or args.method == "auto_cot" else ""

    # Prepare data for multiprocessing
    data_args = [
        (i, data, args, demo, decoder)
        for i, data in enumerate(dataloader) if i >= args.resume_id - 1
    ]

    total = 0
    correct_list = []
    output_results = []

    # Use multiprocessing Pool for parallel execution
    with Pool(processes=min(cpu_count(), args.max_num_worker or cpu_count())) as pool:
        for i, output_line, correct in pool.imap_unordered(process_data, data_args):
            output_results.append(output_line)
            correct_list.append(correct)
            total += 1
            accuracy = (sum(correct_list) / total) * 100
            print(f"{i + 1}st data processed.")
            print(f"Accuracy: {accuracy:.2f}%")
    
    # Write results to output file
    with open(args.output_dir, "a") as wp:
        for result in output_results:
            wp.write(json.dumps(result) + "\n")

    # Final accuracy
    accuracy = (sum(correct_list) / total) * 100
    print(f"Final accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
