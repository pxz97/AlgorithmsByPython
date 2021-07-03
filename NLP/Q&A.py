import os
import time
import json
import random
import datetime
from tqdm import tqdm
import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import DataProcessor, squad_convert_examples_to_features

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


class SquadProcessor(DataProcessor):

    def get_train_examples(self, data_dir, filename=None):

        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir, filename=None):

        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "dev")

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []
                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        answers = qa["answers"][0]
                        answer_text = qa["answers"][0]["text"]
                        start_position_character = qa["answers"][0]["answer_start"]

                    example = ChineseSquardExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                    examples.append(example)
                return examples


class SquadV3Processor(SquadProcessor):
    train_file = "train-v2.0.json"
    dev_file = "dev-v2.0.json"


class ChineseSquardExample:
    def __init__(
            self,
            qas_id,
            question_text,
            context_text,
            answer_text,
            start_position_character,
            title,
            answers=[],
            is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text.replace(" ", "").replace("  ", "").replace(" ", "")
        self.answer_text = ""
        for e in answer_text.replace(" ", "").replace("  ", "").replace(" ", ""):
            self.answer_text += e
            self.answer_text += " "
        self.answer_text = self.answer_text[0: -1]

        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers
        self.doc_tokens = [e for e in self.context_text]
        self.char_to_word_offset = [i for i, e in enumerate(self.context_text)]
        self.start_position = self.context_text.find(answer_text.replace(" ", "").replace("  ", "").replace(" ", ""))
        self.end_position = self.start_position + len(answer_text.replace(" ", "").replace("  ", "").replace(" ", ""))


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def training(train_dataloader, model):
    t0 = time.time()
    total_train_loss = 0
    total_train_accuracy = 0
    model.train()

    for step, batch in enumerate(train_dataloader):

        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print("  Batch {:>5,} of {:>5,}.  Elapsed: {:}.".format(step, len(train_dataloader), elapsed))

        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        start_positions = batch[3].to(device)
        end_positions = batch[4].to(device)

        model.zero_grad()
        loss, start_scores, end_scores = model(input_ids,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               start_positions=start_positions,
                                               end_positions=end_positions)
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_train_loss / len(train_dataloader)
    print("  平均训练损失loss: {0:.2f}".format(avg_train_loss))
    return avg_train_loss


def train_evalution(test_dataloader, model):
    total_eval_loss = 0
    model.eval()

    for batch in test_dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        start_positions = batch[3].to(device)
        end_positions = batch[4].to(device)

        with torch.no_grad():
            loss, start_scores, end_scores = model(input_ids,
                                                   attention_mask=attention_mask,
                                                   token_type_ids=token_type_ids,
                                                   start_positions=start_positions,
                                                   end_positions=end_positions)
        total_eval_loss += loss.item()
    return total_eval_loss, len(test_dataloader)


data_dir = "data/"
processor = SquadV3Processor()
train_data = processor.get_train_examples(data_dir)
dev_data = processor.get_dev_examples(data_dir)

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
model = BertForQuestionAnswering.from_pretrained("hfl/chinese-roberta-wwm-ext")
model.to(device)

seq_length = 1280
query_length = 128






