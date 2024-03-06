import os
import json

import luigi
import law

law.contrib.load("tasks")  # to have the RunOnceTask

from utils.dataset import ParquetDataset

class CoffeaToParquet(law.Task):

    cfg = luigi.Parameter(description="YAML configuration file with input features and features to pad")
    input_file = luigi.Parameter(description="Input Coffea file")
    output_file = luigi.Parameter(description="Output parquet file")
    cat = luigi.Parameter(default="semilep_LHE", description="Event category")

    def read_datasets_definition(self):
        with open(os.path.abspath(self.datasets_definition), "r") as f:
            return json.load(f)
    
    def output(self):
        datasets = self.read_datasets_definition()
        dataset_paths = set()
        for dataset in datasets.values():
            filepath = os.path.abspath(f"{dataset['json_output']}")
            dataset_paths.add(filepath)
            dataset_paths.add(f"{filepath}".replace(".json", "_redirector.json"))
            
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

    cfg = luigi.Parameter(description="YAML configuration file with input features and features to pad")
    input_file = luigi.Parameter(description="Input Coffea file")
    output_file = luigi.Parameter(description="Output parquet file")
    fully_matched = luigi.BoolParameter(default=False, description="Consider only fully matched events")

    def requires(self):
        return CoffeaToParquet.req(self)

    def output(self):
        return law.LocalFileTarget(self.output_file)

    def run(self):
        pass
        
