from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import AutoTokenizer
import sys
import argparse
from setting import ROOT
from transformers import LineByLineTextDataset

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Generate tokens from decoder models')

    parser.add_argument('--train_path', type=str, default='/scratch2/jliu/Generative_replay/knn-transformers/dataset/train/concat/20M.txt',
                        help='Path to the train file')

    parser.add_argument('--val_path', type=str, default='/scratch2/jliu/Generative_replay/knn-transformers/dataset/dev/dev.txt',
                        help='Path to the validation file')

    parser.add_argument('--out_path', type=str, default='/scratch2/jliu/Generative_replay/knn-transformers/model/20M',
                        help='Path to the model dir')

    parser.add_argument('--tokenizer_path', type=str, default='gpt2',
                        help='Path to the bpe tokenizer')
    
    parser.add_argument('--preprocess', default=False,
                        help='whether to preprocess the dataset')

    return parser.parse_args(argv)

# largest size of each block
block_size = 512
model_max_length = 2048

# Configure the decoder-only transformer model
config = GPT2Config(
    vocab_size=60,
    max_position_embeddings=514,
    n_head=8,  # Number of attention heads
    n_layer=6,  # Number of hidden layers
    n_embd=512,  # Hidden size (embedding dimension)
)


def tokenize_data(tokenizer,data_path,block_size:int):
    """tokenize the dataset"""

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=data_path,
        block_size=block_size,
    )
    return dataset

def main(argv):
    # Args parser
    args = parseArgs(argv)
    train_path = args.train_path
    val_path = args.val_path
    out_path = args.out_path
    preprocess = args.preprocess


    print('######################')
    print('Tokenizing the dataset')
    print('######################')

    # preprocess the dataset if necessary
    if preprocess:
        preprocess(train_path)
        preprocess(val_path)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print('Tokenizer loaded')
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    train_dataset = tokenize_data(tokenizer,train_path,block_size)
    val_dataset = tokenize_data(tokenizer, val_path, block_size)

    print('#################')
    print('Loading the model')
    print('#################')
    # Initialize the model with the configured settings
    model = GPT2LMHeadModel(config=config)

    training_args = TrainingArguments(
        output_dir=out_path,
        overwrite_output_dir=True,
        per_gpu_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=False,
        num_train_epochs=20,
        report_to="wandb",  # Enable W&B logging
        logging_dir=out_path,  # Directory for storing logs
        logging_steps=500,  # Log every 500 steps
        evaluation_strategy="steps",  # Evaluate during training at each `eval_steps`
        eval_steps=500,  # Evaluation and early stopping check every 500 steps
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Optional: Specify the metric to use
        greater_is_better=False,  # Optional: Specify if higher metric values are better
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset  # Assuming you have a validation set
    )



    print('##############')
    print('Start training')
    print('##############')

    trainer.train()
    # save the trained model
    trainer.save_model(out_path)
    print(f'Saved the model to {out_path}')


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)