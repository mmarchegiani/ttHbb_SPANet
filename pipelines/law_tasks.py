import os
import json

import luigi
import law

law.contrib.load("tasks")  # to have the RunOnceTask

from utils.dataset.parquet import ParquetDataset
from utils.dataset.h5 import H5Dataset

class CoffeaToParquet(law.Task):

    cfg = luigi.Parameter(description="YAML configuration file with input features and features to pad")
    input_file = luigi.Parameter(description="Input Coffea file")
    output_file = luigi.Parameter(description="Output parquet file")
    cat = luigi.Parameter(default="semilep_LHE", description="Event category")
 
    def output(self):
        return law.LocalFileTarget(self.output_file)
    
    def run(self):
        dataset = ParquetDataset(
            self.input_file,
            self.output_file,
            self.cfg,
            self.cat,
        )
        dataset.save_parquet()

class ParquetToH5(TaskBase):

    cfg = luigi.Parameter(description="YAML configuration file with input features targets and output nodes to save in the h5 file")
    input_file = luigi.Parameter(description="Input parquet file")
    output_file = luigi.Parameter(description="Output h5 file")
    fully_matched = luigi.BoolParameter(default=False, description="Consider only fully matched events")
    no_shuffle = luigi.BoolParameter(default=False, description="If set, do not shuffle the dataset")

    def requires(self):
        return CoffeaToParquet.req(self)

    def output(self):
        return law.LocalFileTarget(self.output_file)

    def run(self):
        dataset = H5Dataset(
            self.input_file,
            self.output_file,
            self.cfg,
            self.fully_matched,
            (not self.no_shuffle)
        )
        dataset.save_h5_all()
        
