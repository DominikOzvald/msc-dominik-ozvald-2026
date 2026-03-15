from torch.utils.data import Dataset, DataLoader
import torch
from Drain import LogParser
from embeddings import LogVocab, CharVocab
import os
import pandas

TEMP_FILE_NAME = "tmp_log.txt"
PARSE_IN_DIR = "./"
PARSE_OUT_DIR = "./"

DEPTH = 4
ST = 0.2
MAX_CHILD = 100

MAX_SIZE = -1
MIN_FREQ = 0

DEFAULT_DICT = {
    "maxChild": MAX_CHILD,
    "st": ST,
    "rex": [],
    "depth": DEPTH,
    "maxSize": MAX_SIZE,
    "minFreq": MIN_FREQ,
}


def extract_with_parse(file_name, parser: LogParser):
    try:
        log_file = open(file_name, "r", encoding="utf-8")
        tmp_file = open(TEMP_FILE_NAME, "w", encoding="utf-8")

        for line in log_file:
            tmp_file.write(line)

        log_file.close()
        tmp_file.close()

        parser.parse(TEMP_FILE_NAME)
        freq_df = pandas.read_csv(TEMP_FILE_NAME + "_templates.csv")
        freq_dict = dict(zip(freq_df["EventTemplate"], freq_df["Occurrences"]))
        log_df = pandas.read_csv(TEMP_FILE_NAME + "_structured.csv")
        log_list = list(log_df["EventTemplate"])

        os.remove(TEMP_FILE_NAME)
        os.remove(TEMP_FILE_NAME + "_templates.csv")
        os.remove(TEMP_FILE_NAME + "_structured.csv")

        return freq_dict, log_list


    except:
        print('skipping file', file_name)
        return {}, []


def extract_raw(file_name):
    logs = []
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                log = line.split(" ", 1)[1]
                logs.append(log)

    except:
        print("Skipping file:", file_name)
    return logs


def form_instances(logs, step: int, frame_size: int):
    instances = []
    for i in range(0, len(logs), step):
        instances.append(logs[i:i + frame_size])
    return instances


def set_kwargs(arg_dict):
    for k in DEFAULT_DICT:
        if k not in arg_dict:
            arg_dict[k] = DEFAULT_DICT[k]


def count_logs(total_frequencies, frequencies):
    for k in frequencies:
        if k in total_frequencies:
            total_frequencies[k] += frequencies[k]
        else:
            total_frequencies[k] = frequencies[k]


def pad_collate_fn(batch):
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0).to(torch.long)

def pad_len_collate_fn(batch):
    lengths = torch.tensor([len(x) for x in batch])
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0).to(torch.long)
    return batch,lengths
def fixed_pad_fn(batch, size=30):
    with torch.no_grad():
        for i in range(len(batch)):
            if (batch[i].shape[-1] < size):
                batch[i] = torch.cat((batch[i], torch.zeros(size - batch[i].shape[-1], )), dim=-1)
            batch[i] = batch[i].unsqueeze(dim=0)
        return torch.cat(batch, dim=0).to(torch.long)


def fixed_pad_fn_factory(size=10):
    return lambda x: fixed_pad_fn(x, size)


class LogDataSet(Dataset):
    def __init__(self, log_dir: str, log_vocab: LogVocab = None, log_format='<DateTime> <Content>', step=1,
                 frame_size=1, **kwargs):

        self.data = []
        self.vocab = log_vocab
        total_frequencies = {}

        set_kwargs(kwargs)

        parser = LogParser(log_format, PARSE_IN_DIR, PARSE_OUT_DIR, maxChild=kwargs["maxChild"], st=kwargs["st"],
                           rex=kwargs["rex"], depth=kwargs["depth"],
                           verbose=False)

        log_files = [file for file in os.listdir(log_dir) if file[-4:] == ".txt"]
        for log_file in log_files:
            frequencies, logs = extract_with_parse(os.path.join(log_dir, log_file), parser)
            self.data += form_instances(logs, step, frame_size)
            if log_vocab is None:
                count_logs(total_frequencies, frequencies)

        if self.vocab is None:
            self.vocab = LogVocab(total_frequencies, max_size=kwargs["maxSize"], min_freq=kwargs["minFreq"])

    def __getitem__(self, item):
        log = self.data[item]
        return self.vocab.encode(log)

    def __len__(self):
        return len(self.data)


class LogCharDataSet(Dataset):

    def __init__(self, log_dir: str, step=1, frame_size=1):
        super().__init__()
        self.data = []
        self.vocab = CharVocab()
        log_files = [file for file in os.listdir(log_dir) if file[-4:] == ".txt"]
        for log_file in log_files:
            logs = extract_raw(os.path.join(log_dir, log_file))
            for i in range(0, len(logs), step):
                frame = logs[i:i + frame_size]

                instance = "".join(frame)
                if len(instance) > 1:
                    self.data.append(instance[:200])

    def __getitem__(self, item):
        log = self.data[item]
        return self.vocab.encode(log)

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    step_size = 1
    frame_size = 1
    data_set = LogCharDataSet("../test_data", frame_size=frame_size, step=step_size)
    x = data_set[0]
    print(data_set[0].shape, data_set[1].shape, data_set[2].shape)
#     re = [r"^[\.s]+$"]
#     data_set = LogDataSet("../test_data", minFreq=2, step=step_size, frame_size=frame_size, rex=re)
#     loader = DataLoader(dataset=data_set, shuffle=False, batch_size=2, collate_fn=fixed_pad_fn_factory(frame_size))
#     matrix = create_embedding_matrix(data_set.vocab,dim=5)
#     print(data_set.vocab.str2int)
#     batch = matrix(next(iter(loader)))
