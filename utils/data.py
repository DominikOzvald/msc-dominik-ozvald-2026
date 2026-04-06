import os

import pandas
import torch
import torch.nn.functional as F
from Drain import LogParser

TEMP_FILE_NAME = "tmp_log.txt"


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
                log = line.split(" ", 1)
                if len(log) > 1:
                    logs.append(log[1])

    except Exception as e:
        print("Skipping file:", file_name, e)
    return logs


def form_instances(logs, step: int, frame_size: int):
    instances = []
    for i in range(0, len(logs), step):
        instances.append(logs[i:i + frame_size])
    return instances


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
    return batch, lengths


def pad_frame_collate_fn(batch):
    frame_lengths = torch.tensor([len(frame) for frame in batch])
    padded_subframes, subframe_lengths = zip(*[pad_len_collate_fn(frame) for frame in batch])
    return padded_subframes, frame_lengths


def fixed_pad_fn(batch, size=30):
    lengths = torch.tensor([len(x) for x in batch])
    with torch.no_grad():
        data = [ F.pad(x,[0,size-x.size(0)],value=0) for x in batch ]
        data = torch.stack(data,dim=0)
    return data,lengths



def fixed_pad_fn_factory(size=10):
    return lambda x: fixed_pad_fn(x, size)


def separate_last_log(data, masks,n_steps:int = 1):
    # index = torch.searchsorted(masks, torch.ones((masks.size(0),1), device=masks.device)) - 1
    # tgt = data[torch.arange(data.size(0), device=data.device), index.view(-1)]
    with torch.no_grad():
        masks_tmp = []
        data_tmp = []
        for i, mask in enumerate(masks):
            if mask.max() == 0:
                masks_tmp.append(mask.unsqueeze(0))
                data_tmp.append(data[i].unsqueeze(0))
        data = torch.cat(data_tmp, dim=0)
        masks = torch.cat(masks_tmp, dim=0)
        tgt = data[:, -n_steps:, :]
        z = data[:, :-n_steps, :]
        masks = masks[:, :-n_steps]
        # for i, length in enumerate(index.view(-1)):
        #     if length < data.size(1) - 1:
        #         masks[i, length] = 1
    return z, tgt, masks
