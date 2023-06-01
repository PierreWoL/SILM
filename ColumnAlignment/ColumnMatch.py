import math
import os

import numpy as np
import torch
from ColumnMatchingClassifier import ColumnMatchingClassifier as Classifier
from transformers import TrainingArguments
from argparse import Namespace
from Dataset_dict import create_column_pairs
from operator import itemgetter

class ColumnMatch:

    def __init__(self,
                 train_path,
                 hp: Namespace):
        self.train_path = train_path
        torch.cuda.empty_cache()
        self.bert_trainer = Classifier.from_hp(train_path, hp)
        epoch_steps = len(self.bert_trainer.train) // hp.batch  # total steps of an epoch
        logging_steps = int(epoch_steps * 0.02) if int(epoch_steps * 0.02) > 0 else 5
        eval_steps = 5 * logging_steps
        training_args = TrainingArguments(
            output_dir=hp.output_dir,
            num_train_epochs=hp.num_epochs,
            per_device_train_batch_size=hp.batch,
            per_device_eval_batch_size=hp.batch,
            weight_decay=0.01,
            logging_steps=logging_steps,
            logging_dir=f"{hp.output_dir}/tb",
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            do_train=True,
            do_eval=True,
            save_steps=eval_steps,
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="accuracy",
            greater_is_better=True,
        )

        self.bert_trainer.train(train_args=training_args, do_fine_tune=hp.fine_tune)
        if hp.fine_tune:
            self.bert_trainer.trainer.save_model(
                output_dir=os.path.join(hp.output_dir, "FT_columnMatch")
            )
            print("fine-tuning done, fine-tuned model saved.")
        else:
            print("pretrained or fine-tuned model loaded.")

        self.bert_trainer.model.eval()
        self.device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.bert_trainer.model.to(self.device)

        self.tokenize = lambda x: self.bert_trainer.tokenizer(
            x, max_length=hp.max_length, truncation=True, padding=True, return_tensors="pt"
        )
        softmax = torch.nn.Softmax(dim=1)
        self.classifier = lambda x: softmax(self.bert_trainer.model(**x).logits)[:, 1]

    def score(self,  hp: Namespace):
        r"""Score the samples with the classifier.

            Args:
                hp: parameters
        """
        samples, samples_mapping = create_column_pairs(hp.eval_path)
        if len(samples)!=0 and len(samples_mapping)!=0:
            sample_size = len(samples)
            scores = np.zeros(sample_size)
            batch_num = math.ceil(sample_size / hp.eval_batch_size)
            for i in range(batch_num):
                j = (i + 1) * hp.eval_batch_size \
                    if (i + 1) * hp.eval_batch_size <= sample_size else sample_size
                inputs = self.tokenize(samples[i * hp.eval_batch_size:j])
                inputs.to(self.device)
                with torch.no_grad():
                    batch_scores = self.classifier(inputs)
                scores[i * hp.eval_batch_size:j] = batch_scores.cpu().numpy()
            results = {samples_mapping[i]: scores[i] for i in range(len(samples_mapping)) if scores[i]  != 0.0}
            print(results)
            sort_results = dict(sorted(results.items(), key=itemgetter(1), reverse=True))
            return results
        else:
            print("Wrong Path!")

