from torch.utils.data import Dataset


class CircularBufferDataset(Dataset):

    def __init__(self, size):
        self.size = size
        self.buffer = list()
        self.ptr = 0
        self.filled = False

    def __getitem__(self, index: int):
        return self.buffer[index]

    def __len__(self) -> int:
        return self.size if self.filled else self.ptr

    def add_items(self, items):
        for item in items:
            if self.filled:
                self.buffer[self.ptr] = item
                self.ptr = (self.ptr + 1) % self.size
            else:
                self.buffer.append(item)
                self.ptr += 1
                if self.ptr == self.size:
                    self.ptr = 0
                    self.filled = True
