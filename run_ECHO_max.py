'''
Adapted from https://github.com/kojima-takeshi188/zero_shot_cot
'''

import argparse
from utils import *
import random
from tqdm import tqdm

def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    
    fix_seed(args.random_seed)
    
    print("OPENAI_API_KEY:")
    print(os.getenv("OPENAI_API_KEY")[0:5] + '**********')
    
    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder()
            
    # Let's start
    

    rcd = -1
    rcd_max_n = -1

    for nc in range(32, 8 - 1 , -1):
        total_num_token = 0
        token_limit = 4096
        token_reserved = 512
        with open(args.demo_path + '_' + str(nc), 'r', encoding="utf-8") as f:
            json_data = json.load(f)
            json_data = json_data["demo"]

            for line in json_data:
                total_num_token += num_tokens_from_string(line["question"])
                total_num_token += num_tokens_from_string(line["rationale"])
                
        if total_num_token > token_limit - token_reserved:
            continue
        else:
            with open(args.demo_path + '_' + str(nc), 'r', encoding="utf-8") as f:
                json_data = json.load(f)
                json_data = json_data["demo"]
                if len(json_data) > rcd_max_n:
                    rcd_max_n = len(json_data)
                    rcd = nc
    
    print("rcd: " + str(rcd), "rcd_max_n: " + str(rcd_max_n))

    while True:        
        try:
            len_list = []
            # if still have trouble, try to use the smaller one
            print("---------------rcd: " + str(rcd) + "---------------")
            if rcd == -1:
                break
            x, z, y, g = [], [], [], []
            inertia = 0
            with open(args.demo_path + '_' + str(rcd), 'r', encoding="utf-8") as f:
                json_data = json.load(f)
                inertia = json_data["inertia"]
                json_data = json_data["demo"]
                for line in json_data:
                    len_list.append(num_tokens_from_string(line["rationale"]))
                    x.append(line["question"])
                    z.append(line["rationale"])
                    y.append(line["pred_ans"])
                    g.append(line["gold_ans"])
                
            avg = np.mean(len_list)
            var = np.std(len_list)
            # alpha = 0.01
            # iters = alpha * var * np.log(inertia) * np.log(avg) / np.log(len(x) + 1)
            iters = args.iter

            print("avg:" + str(avg) + " number of example:" + str(len(x))+ " std:" + str(var) + " iters: " + str(iters))
   
            
            index_list = list(range(len(x)))

            for p in range(iters):
                index_list_new = index_list[:]
                for q in tqdm(range(len(x)), total=len(x)):
                    # print("in" + str(i))
                    i = random.choice(index_list_new)
                    index_list_new.remove(i)
                    remaining_list = index_list[:i] + index_list[i+1:]
                    # shuffle remaining_list
                    random.shuffle(remaining_list)
                    # create demo text for this instance
                    demo_text = ""
                    for r in remaining_list:
                        if args.direct_answer_trigger_for_fewshot not in z[r]:
                            demo_text += x[r] + " " + z[r] + " " + args.direct_answer_trigger_for_fewshot + " " + y[r] + ".\n\n"
                        else:
                            demo_text += x[r] + " " + z[r] + "\n\n"
                    question = demo_text + x[i] + " " + args.cot_trigger
                    max_length = args.max_length_cot
                    z_new = decoder.decode(args, question, max_length)
                    # Clensing of predicted answer ...
                    y_new = answer_cleansing(args, z_new)
                    z[i] = args.cot_trigger + " "  + z_new[:z_new.find(args.direct_answer_trigger_for_fewshot)].strip()
                    y[i] = y_new.strip()

            break

        except Exception as e:
            print(str(rcd) + " error" + str(e))
            rcd -= 1
            print("---------------rcd: " + str(rcd) + "---------------")
        


    task = args.dataset

    if task == "aqua" or task == "last_letters":
        num_clusters = 4
    elif task == "commonsensqa":
        num_clusters = 7
    elif task == "strategyqa":
        num_clusters = 6
    else:
        num_clusters = 8

    demos = []
    for i in range(num_clusters):
        demo_element = {
                        "question": x[i],
                        "rationale": z[i],
                        "pred_ans": y[i],
                        "gold_ans": g[i],
                    }
        demos.append(demo_element)

    demos = {"demo": demos}

    with open(args.output_dir, 'w', encoding="utf-8") as write_f:
        json.dump(demos, write_f, indent=4, ensure_ascii=False)
    


    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="multiarith", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters"], help="dataset used for experiment"
    )
    parser.add_argument(
        "--demo_path", type=str, default="demos/multiarith", help="pre-generated demos used for experiment"
    )
    parser.add_argument(
        "--model", type=str, default="gpt3-xl", choices=["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl", "code-davinci-002", "gpt-3.5-turbo-0301", "gpt-3.5-turbo","gpt-3.5-turbo-16k-0613"], help="model used for decoding. Note that 'gpt3' are the smallest models."
    )
    parser.add_argument(
        "--output_dir", type=str, default="CAT_demos/multiarith", help="output directory"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=512, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--iter", type=int, default=4, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=0.1, help="sleep between runs to avoid exceeding the rate limit of openai api"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="temperature for GPT-3"
    )

    parser.add_argument(
        "--method", type=str, default="auto_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot"], help="method"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."
    
    return args

if __name__ == "__main__":
    main()