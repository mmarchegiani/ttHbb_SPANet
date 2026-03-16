from tthbb_spanet.lib.dataset.base import Dataset
from tthbb_spanet.lib.dataset.spanet_dataset import SPANetDataset
from tthbb_spanet.lib.dataset.dctr_dataset import DCTRDataset

# ParquetDataset depends on coffea; keep it optional for scripts
# that do not need parquet conversion.
try:
    from tthbb_spanet.lib.dataset.parquet import ParquetDataset
except ModuleNotFoundError:
    ParquetDataset = None