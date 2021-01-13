import os

root_dir = "results/eval-all/"
eval_list = ["anli-full-dev", "anli-full-test", "anli-r1-dev", "anli-r1-test", "anli-r2-dev", "anli-r2-test",
             "anli-r3-dev", "anli-r3-test", "mnli-dev", "mnli-mm-dev", "mnli-bert-adv", "mnli-mm-bert-adv",
             "mnli-roberta-adv", "mnli-mm-roberta-adv", "snli-bert-adv", "snli-dev", "snli-roberta-adv"]
for file in os.listdir(root_dir):
    cur_path = os.path.join(root_dir, file)
    print(file)

    with open(os.path.join(cur_path, "eval_results.txt")) as f:
        lines = f.readlines()
        results = {}
        for i in range(0, len(lines), 5):
            name = lines[i]
            acc = round(float(lines[i+2].split()[-1]) * 100, 1)

            for eval_item in eval_list:
                if eval_item in name:
                    results[eval_item] = acc

        print("=====For ANLI table=============")
        print(f"{results['anli-r1-dev']} & {results['anli-r2-dev']} & {results['anli-r3-dev']} & {results['anli-full-dev']} & "
              f"{results['anli-r1-test']} & {results['anli-r2-test']} & {results['anli-r3-test']} & {results['anli-full-test']}")

        print("=====For TextFooler table=============")
        print(
            f"{results['snli-dev']} & {results['mnli-dev']}/{results['mnli-mm-dev']} & "
            f"{results['snli-bert-adv']} & {results['mnli-bert-adv']}/{results['mnli-mm-bert-adv']} & "
            f"{results['snli-roberta-adv']} & {results['mnli-roberta-adv']}/{results['mnli-mm-roberta-adv']} ")

