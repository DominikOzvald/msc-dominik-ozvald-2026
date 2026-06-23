import torch
from torch.utils.data import Dataset
from utils.embeddings import CharVocab
from utils.data import extract_raw, extract_tagged
import os
import bisect
import torch.nn.functional as F


class TransformerDataset(Dataset):
    def __init__(self, log_dir: str, step=1, frame_size=1, max_len=200, ):
        super().__init__()
        self.data = []
        self.vocab = CharVocab()
        self.step = step
        self.frame_size = frame_size
        self.file_starts = []
        self.max_len = max_len
        self.file_frames = []
        log_files = [file for file in os.listdir(log_dir) if file[-4:] == ".txt"]
        total_num_frames = 0
        for log_file in log_files:
            self.file_frames.append(total_num_frames)
            self.file_starts.append(len(self.data))
            logs = extract_raw(os.path.join(log_dir, log_file))
            self.data += [log[:max_len] for log in logs]
            num_frames = len(logs) // self.step
            if len(logs) % self.step:
                num_frames += 1

            total_num_frames += num_frames
        self.length = total_num_frames

    def _frame_ends(self, item):
        frame_start_file = bisect.bisect_right(self.file_frames, item) - 1

        frame_start = self.file_starts[frame_start_file] + (item - self.file_frames[frame_start_file]) * self.step
        frame_end = frame_start + self.frame_size
        frame_end_file = bisect.bisect_right(self.file_starts, frame_end) - 1
        if frame_start_file != frame_end_file:
            frame_end = self.file_starts[frame_start_file + 1]
        return frame_start, frame_end

    def __getitem__(self, item):
        frame_stat, frame_end = self._frame_ends(item)
        logs = self.data[frame_stat:frame_end]
        with torch.no_grad():
            enc_logs = [self.vocab.encode(log) for log in logs]
            lengths = torch.Tensor([len(log) for log in enc_logs])
            frame = torch.stack([F.pad(log, (0, self.max_len - log.size(0)), value=0) for log in enc_logs]).to(
                torch.long)
            frame_len = frame.size(0)
            if frame_len < self.frame_size:
                frame = F.pad(frame, (0, 0, 0, self.frame_size - frame_len), value=0)
                lengths = torch.cat([lengths, torch.ones(self.frame_size - frame_len)])
                mask = torch.cat([torch.zeros(frame_len), torch.ones(self.frame_size - frame_len)])
            else:
                mask = torch.zeros(frame_len)
        return frame, lengths, mask

    def __len__(self):
        return self.length


class DummyCharDataSet(Dataset):
    def __init__(self, log_dir: str = None, max_in_len: int = 200):
        super().__init__()
        self.data = []
        self.vocab = CharVocab()
        self.max_in_len = max_in_len
        if log_dir:
            log_files = [file for file in os.listdir(log_dir) if file[-4:] == ".txt"]
            for log_file in log_files:
                self.add_from_file(os.path.join(log_dir, log_file))

    def add_from_file(self, file_path):
        pairs = extract_tagged(file_path, self.max_in_len)
        logs, tags = zip(*pairs)
        for log in logs:
            if log not in self.data:
                self.data.append(log[:self.max_in_len])

    def get_unencoded(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.vocab.encode(self.data[item])


class DummyLogDataSet(Dataset):
    def __init__(self, log_dir: str = None, step: int = 5, frame_size: int = 30, max_in_len: int = 200,
                 pad_tag: int = 0):
        super().__init__()
        self.step = step
        self.frame_size = frame_size
        self.vocab = CharVocab()
        self.data = []
        self.files = []
        self.max_in_len = max_in_len
        self.pad_tag = pad_tag
        if log_dir:
            log_files = [file for file in os.listdir(log_dir) if file[-4:] == ".txt"]
            for log_file in log_files:
                self.add_from_file(os.path.join(log_dir, log_file))

    def add_from_file(self, file_name):
        pairs = extract_tagged(file_name, self.max_in_len)
        file_start = len(self.data)
        num_frames = 0
        for i in range(0, len(pairs), self.step):
            self.data.append(pairs[i:i + self.frame_size])
            num_frames += 1
        self.files.append((file_name, file_start, num_frames))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        logs, tags = zip(*self.data[item])
        with torch.no_grad():
            lengths = torch.Tensor([len(log) for log in logs])
            tags = torch.Tensor(tags)
            enc_logs = [self.vocab.encode(log) for log in logs]
            padded_log = [F.pad(log, (0, self.max_in_len - log.size(0)), value=0) for log in enc_logs]
            frame = torch.stack(padded_log, dim=0)
            frame_len = frame.size(0)
            if frame_len < self.frame_size:
                frame = F.pad(frame, (0, 0, 0, self.frame_size - frame_len), value=0)
                lengths = torch.cat([lengths, torch.ones(self.frame_size - frame_len)])
                tags = torch.cat([tags, self.pad_tag * torch.ones(self.frame_size - frame_len)])
                mask = torch.cat([torch.zeros(frame_len), torch.ones(self.frame_size - frame_len)])
            else:
                mask = torch.zeros(frame_len)
            tags = tags.to(torch.long)
        return frame, lengths, mask, tags
