import pandas as pd
from torch.utils.data.dataset import Dataset


class MyDataset(Dataset):

    def __init__(self, datas: pd.DataFrame) -> None:
        super().__init__()
        self.datas = datas

    def __getitem__(self, index: int):
        x = self.datas.iloc[index]['data']
        y = self.datas.iloc[index]['label']
        id = self.datas.iloc[index]['id']
        return x, y, id

    def __len__(self):
        return len(self.datas)

    def get_labels(self, ):
        return self.datas['label'].tolist()

