
import os

root_dir = "results/eval-all/"
eval_list = [
    "anli-full-dev", "anli-full-test", "anli-r1-dev", "anli-r1-test", "anli-r2-dev", "anli-r2-test",
    "anli-r3-dev", "anli-r3-test", "mnli-dev", "mnli-mm-dev", "mnli-bert-adv", "mnli-mm-bert-adv",
    "mnli-roberta-adv", "mnli-mm-roberta-adv", "snli-bert-adv", "snli-dev", "snli-roberta-adv"
]

def extract_accuracy(lines):
    return {eval_item: round(float(lines[i+2].split()[-1]) * 100, 1)
            for i in range(0, len(lines), 5)
            for eval_item in eval_list if eval_item in lines[i]}

def print_table(results, label):
    dev_items = [results.get(f"{eval_item}-dev", '-') for eval_item in eval_list]
    test_items = [results.get(f"{eval_item}-test", '-') for eval_item in eval_list]

    print(f"=====For {label} table=============")
    print(" & ".join(map(str, dev_items + test_items)))

for file in os.listdir(root_dir):
    cur_path = os.path.join(root_dir, file)
    print(file)

    with open(os.path.join(cur_path, "eval_results.txt")) as f:
        lines = f.readlines()
        results = extract_accuracy(lines)

    print_table(results, "ANLI")
    print_table(results, "TextFooler")

