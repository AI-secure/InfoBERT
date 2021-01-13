# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch

from transformers import AutoConfig, AutoTokenizer, EvalPrediction
from models.modeling_auto import AutoModelForSequenceClassification
from MI_estimators import CLUB, CLUBv2, InfoNCE
from datasets.anli import GlueDataset, GlueDataTrainingArguments as DataTrainingArguments
from processors.anli import glue_output_modes, glue_tasks_num_labels, glue_compute_metrics
from local_robust_trainer import Trainer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from advtraining_args import TrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    load: Optional[str] = field(
        default=None, metadata={"help": "the path to load pretrained models"}
    )
    beta: float = field(
        default=0, metadata={"help": "the regularization term"}
    )
    version: float = field(
        default=-1, metadata={"help": "version of MI estimator"}
    )


def main():
    # See all possible arguments in src/transformers/advtraining_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
            and training_args.local_rank in [-1, 0]
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    root_dir = training_args.output_dir
    if not os.path.exists(root_dir) and training_args.local_rank in [-1, 0]:
        os.mkdir(root_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[
            logging.FileHandler(os.path.join(training_args.output_dir, "log.txt")),
            logging.StreamHandler()
        ] if training_args.local_rank in [-1, 0] else [logging.StreamHandler()]
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        output_hidden_states=True,
        attention_probs_dropout_prob=training_args.attention_probs_dropout_prob,
        hidden_dropout_prob=training_args.hidden_dropout_prob
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    # take the embedding of the whole sentence as varaible y
    if model_args.version >= 0:
        if model_args.version == 0:
            mi_upper_estimator = CLUB(config.hidden_size * data_args.max_seq_length, config.hidden_size,
                                beta=model_args.beta).to(training_args.device)
            mi_upper_estimator.version = 0
            mi_estimator = None
        elif model_args.version == 1:
            mi_upper_estimator = CLUB(config.hidden_size, config.hidden_size, beta=model_args.beta).to(training_args.device)
            # mi_estimator = CLUB(config.hidden_size, config.hidden_size,  beta=model_args.beta)
            mi_upper_estimator.version = 1
            mi_estimator = None
        elif model_args.version == 2 or model_args.version == 3:
            mi_upper_estimator = CLUBv2(config.hidden_size, config.hidden_size, beta=model_args.beta).to(training_args.device)
            mi_upper_estimator.version = model_args.version
            mi_estimator = None
        elif model_args.version == 4:
            mi_estimator = InfoNCE(config.hidden_size, config.hidden_size).to(training_args.device)
            mi_upper_estimator = None
        elif model_args.version == 5:
            mi_estimator = InfoNCE(config.hidden_size, config.hidden_size).to(training_args.device)
            mi_upper_estimator = CLUBv2(config.hidden_size, config.hidden_size, beta=model_args.beta).to(training_args.device)
            mi_upper_estimator.version = 2
        elif model_args.version == 6:
            mi_estimator = InfoNCE(config.hidden_size, config.hidden_size).to(training_args.device)
            mi_upper_estimator = CLUBv2(config.hidden_size, config.hidden_size, beta=model_args.beta).to(training_args.device)
            mi_upper_estimator.version = 3
    else:
        mi_estimator = None
        mi_upper_estimator = None

    # Get datasets
    train_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    )
    eval_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="dev")
        if training_args.do_eval
        else None
    )
    test_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="test")
        if training_args.do_predict
        else None
    )

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    if model_args.load is not None:
        print(model_args.load)
        model.load_state_dict(torch.load(os.path.join(model_args.load, "pytorch_model.bin")))
        if mi_estimator:
            mi_estimator.load_state_dict(torch.load(os.path.join(model_args.load, "mi_estimator.bin")))
        logger.info(f"Load successful from {model_args.load}")

    if os.path.isdir(model_args.model_name_or_path):
        if mi_estimator:
            mi_estimator.load_state_dict(torch.load(os.path.join(model_args.model_name_or_path, "mi_estimator.bin")))
            logger.info(f"Load mi estimator successful from {model_args.model_name_or_path}")


    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
        mi_estimator=mi_estimator,
        mi_upper_estimator=mi_upper_estimator
    )
    trainer.tokenizer = tokenizer

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

            if mi_estimator:
                torch.save(mi_estimator.state_dict(), os.path.join(training_args.output_dir, "mi_estimator.bin"))

            torch.save(trainer.eval_hist, os.path.join(training_args.output_dir, 'eval_hist.bin'))

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev")
            )
        if 'anli' in data_args.task_name:
            eval_datasets.append(
                    GlueDataset(data_args, tokenizer=tokenizer, mode="test")
            )

        if data_args.task_name == 'anli-full' or data_args.task_name == 'anli-part':
            eval_tasks = ["anli-r1", "anli-r2", "anli-r3", "mnli", "mnli-mm", "snli",
                          "mnli-bert-adv", "mnli-mm-bert-adv", "snli-bert-adv",
                          "mnli-roberta-adv", "mnli-mm-roberta-adv", "snli-roberta-adv"]
            for task in eval_tasks:
                if "mnli" in task and 'adv' not in task:
                    task_data_dir = os.path.join(data_args.data_dir, "MNLI")
                elif "snli" == task and 'adv' not in task:
                    task_data_dir = os.path.join(data_args.data_dir, "SNLI")
                else:
                    task_data_dir = data_args.data_dir
                task_data_args = dataclasses.replace(data_args, task_name=task, data_dir=task_data_dir)
                eval_datasets.append(
                    GlueDataset(task_data_args, tokenizer=tokenizer, mode="dev")
                )
                if 'anli' in task:
                    eval_datasets.append(
                        GlueDataset(task_data_args, tokenizer=tokenizer, mode="test")
                    )

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)
            # eval_result = trainer.evaluate_mi(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}-{eval_dataset.mode}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info(f"***** Eval results {eval_dataset.args.task_name}-{eval_dataset.mode} *****")
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

        if trainer.eval_hist:
            best_eval = trainer.eval_hist[0]
            for eval in trainer.eval_hist:
                if eval['eval_acc'] > best_eval['eval_acc']:
                    best_eval = eval
            output_eval_file = os.path.join(
                training_args.output_dir, f"best_eval_results_{data_args.task_name}_.txt"
            )
            with open(output_eval_file, "w") as writer:
                logger.info("***** Best Eval results {} *****".format(data_args.task_name))
                for key, value in best_eval.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            del trainer.model
            torch.cuda.empty_cache()
            # re-evaluate the best parameters
            checkpoint = os.path.join(training_args.output_dir, f"checkpoint-{best_eval['step']}")
            # trainer.model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model.bin')))
            trainer.model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(training_args.device)
            logger.info(f"successfully load from {checkpoint}")

            for eval_dataset in eval_datasets:
                trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
                # eval_result = trainer.evaluate(eval_dataset=eval_dataset)
                eval_result = trainer.evaluate(eval_dataset=eval_dataset)

                output_eval_file = os.path.join(
                    training_args.output_dir, f"best_eval_results_{eval_dataset.args.task_name}-{eval_dataset.mode}.txt"
                )
                if trainer.is_world_master():
                    with open(output_eval_file, "w") as writer:
                        logger.info(f"***** Best Eval results {eval_dataset.args.task_name}--{eval_dataset.mode} *****")
                        for key, value in eval_result.items():
                            logger.info("  %s = %s", key, value)
                            writer.write("%s = %s\n" % (key, value))

            # # double eval to test whether there is stocahsticasty during evaluation
            # for eval_dataset in eval_datasets:
            #     trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            #     # eval_result = trainer.evaluate(eval_dataset=eval_dataset)
            #     eval_result = trainer.evaluate(eval_dataset=eval_dataset)
            #
            #     output_eval_file = os.path.join(
            #         training_args.output_dir, f"best_eval_results_{eval_dataset.args.task_name}.txt"
            #     )
            #     if trainer.is_world_master():
            #         with open(output_eval_file, "w") as writer:
            #             logger.info("***** Best Eval results {} *****".format(eval_dataset.args.task_name))
            #             for key, value in eval_result.items():
            #                 logger.info("  %s = %s", key, value)
            #                 writer.write("%s = %s\n" % (key, value))

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
            )

        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()