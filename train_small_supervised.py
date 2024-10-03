import os
import sys
import torch
import pandas as pd
import numpy as np
from functools import partial
from dataclasses import dataclass, field
from datasets import (
    Dataset,
    DatasetDict,
    Value,
    Sequence,
    concatenate_datasets
)
# from tokenization_small100 import SMALL100Tokenizer
# import evaluate
import transformers
from transformers import (
    HfArgumentParser,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    default_data_collator,
    
)
import sacrebleu


def get_save_steps(
    num_samples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    max_save_checkpoint: int,
) -> int:
    total_training_steps = num_samples // batch_size // gradient_accumulation_steps * num_epochs
    return total_training_steps // max_save_checkpoint


# Metric
# metric = evaluate.load("sacrebleu")


def cast_columns_to_int32(dataset_object: Dataset) -> Dataset:
    new_features = dataset_object.features.copy()
    for key in new_features.keys():
        new_features[key] = Sequence(Value("int32"))

    dataset_object = dataset_object.cast(new_features)
    return dataset_object


def preprocess_dataset(examples, tokenizer, source_lang, target_lang):
    # inputs = [example[source_lang] for example in examples["translation"]]
    # targets = [example[target_lang] for example in examples["translation"]]
    inputs = []
    targets = []
    for example in examples["translation"]:
        inputs.append(example[source_lang])
        targets.append(example[target_lang])

    model_inputs = tokenizer(inputs, text_target=targets, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    
    return model_inputs


def create_and_process_translated_dataset(dataframe: pd.DataFrame, tokenizer, source_lang, target_lang, num_processors: int) -> Dataset:
    dataframe = dataframe.dropna()
    ids_and_translations = []
    for i in range(len(dataframe)):
        ids_and_translations.append({
            "id": i, 
            "translation": {
                source_lang: dataframe["text"].iloc[i],
                target_lang: dataframe["prediction"].iloc[i]
            }
        }) 
    format_translation_dataframe = pd.DataFrame(ids_and_translations)
    translation_dataset = Dataset.from_pandas(format_translation_dataframe)
    tokenized_dataset = translation_dataset.map(
                            partial(preprocess_dataset, 
                                    source_lang=source_lang,
                                    target_lang=target_lang,
                                    tokenizer=tokenizer),
                            load_from_cache_file=False,
                            num_proc=num_processors,
                            batched=True,
                        )
    
    # tokenized_dataset = cast_columns_to_int32(tokenized_dataset)
    return tokenized_dataset


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    # result = {"bleu": result["score"]}
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    result = {"bleu": bleu.score}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

@dataclass
class DataTrainingArguments:
    # data_csv_path: str = field(default="", metadata={
    #                            "help": "Translation Dataset CSV path"})
    src_to_trg_data_csv_path: str = field(default="", metadata={
                                 "help": "Translation Dataset from source to target CSV path"})

    eval_data_csv_path: str = field(default="", metadata={
                                    "help": "Evaluation Dataset CSV path"})

    trg_to_src_data_csv_path: str = field(default="", metadata={
                                 "help": "Translation Dataset from target to source CSV path"})
    checkpoint_dir: str = field(
        default="../checkpoints/", metadata={"help": "Checkpoint path"})
    num_processors: int = field(
        default=8, metadata={"help": "Number of processors"})
    pad_to_max_length: bool = field(default=False, metadata={
                                    "help": "Pad to max length"})
    ignore_pad_token_for_loss: bool = field(
        default=True, metadata={"help": "Ignore pad token for loss"})


@dataclass
class ExtraTrainingArguments:
    max_save_checkpoint: int = field(
        default=1, metadata={"help": "only save X checkpoints over training steps"})


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    # Set to False for deterministic backends (i.e. reproducable results)
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(42)
    transformers.set_seed(42)

    parser = HfArgumentParser(
        (DataTrainingArguments, ExtraTrainingArguments, Seq2SeqTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, extra_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args, extra_args, training_args = parser.parse_args_into_dataclasses()

    # Load the tokenizer.
    tokenizer = M2M100Tokenizer.from_pretrained(
        data_args.checkpoint_dir, use_fast=True, tgt_lang="th")
    

    # Initialize the model.
    # model = AutoModelForSeq2SeqLM.from_pretrained(data_args.checkpoint_dir, from_tf=True)
    model = M2M100ForConditionalGeneration.from_pretrained(data_args.checkpoint_dir)

    print(f"No. of parameters: {model.num_parameters()}")

    # Set the data collator.
    label_pad_token_id = - 100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if (training_args.fp16 or training_args.bf16) else None,
        )

    # EN-TH Translation. # TODO: Not sure that we could skip EN-TH translation or not and merge it to just one process, TH-EN translation or not.
    assert data_args.trg_to_src_data_csv_path is not None, print(
        'Must provide the csv file for training data.')
    trg_to_src_translated_dataframe = pd.read_csv(data_args.trg_to_src_data_csv_path, low_memory=False)
    trg_to_src_tokenized_dataset = create_and_process_translated_dataset(
        dataframe=trg_to_src_translated_dataframe,
        tokenizer=tokenizer,
        source_lang="en",
        target_lang="th",
        num_processors=data_args.num_processors,
    )
    # TH-EN Translation.
    assert data_args.src_to_trg_data_csv_path is not None, print(
        'Must provide the csv file for training data.')
    src_to_trg_translated_dataframe = pd.read_csv(data_args.src_to_trg_data_csv_path, low_memory=False)
    src_to_trg_tokenized_dataset = create_and_process_translated_dataset(
        dataframe=src_to_trg_translated_dataframe,
        tokenizer=tokenizer,
        source_lang="th",
        target_lang="en",
        num_processors=data_args.num_processors,
    )
    
    assert data_args.eval_data_csv_path is not None, print(
        'Must provide the csv file for evaluation data.')
    eval_dataframe = pd.read_csv(data_args.eval_data_csv_path, low_memory=False)
    eval_tokenized_dataset = create_and_process_translated_dataset(
        dataframe=eval_dataframe,
        tokenizer=tokenizer,
        source_lang="en",
        target_lang="th",
        num_processors=data_args.num_processors,
    )

    tokenized_dataset =  DatasetDict()
    tokenized_dataset["train"] = concatenate_datasets(
        [trg_to_src_tokenized_dataset, src_to_trg_tokenized_dataset])
    
    tokenized_dataset["test"] = eval_tokenized_dataset
    tokenized_dataset["train"] = tokenized_dataset["train"].shuffle(seed=42)

    print(tokenized_dataset)
    
    training_args.load_best_model_at_end = True
    training_args.predict_with_generate = True
    training_args.save_total_limit = extra_args.max_save_checkpoint
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=True)
    # Save the model
    trainer.save_model("trained_models/SMALL-100-model-save(2)")
