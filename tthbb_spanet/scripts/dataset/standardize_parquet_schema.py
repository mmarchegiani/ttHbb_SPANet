# Option 1: Standardize schema when reading
import os, sys
import pyarrow as pa
import pyarrow.parquet as pq

BASE_FOLDER="/eos/home-m/mmarcheg/ttHbb/dask_jobs/ntuples_dctr/total/ntuples"
OUTPUT_FOLDER="/eos/home-m/mmarcheg/ttHbb/dask_jobs/ntuples_dctr/total/recovered_ntuples"

# Read the table with schema modification
def standardize_schema(table):
    schema = table.schema
    new_fields = []
    
    for field in schema:
        if pa.types.is_list(field.type):
            # Standardize the list field name to 'item'
            new_list_type = pa.list_(field.type.value_type)
            new_fields.append(pa.field(field.name, new_list_type))
        else:
            new_fields.append(field)
    
    new_schema = pa.schema(new_fields)
    return table.cast(new_schema)

# Use it when reading your files
error_logfile = "/afs/cern.ch/work/m/mmarcheg/error_parquet_metadata.log"
with open(error_logfile, 'r') as f:
    datasets = f.read().splitlines()

for dataset in datasets:
    print(f"Working on dataset: {dataset}")
    dataset_folder = os.path.join(BASE_FOLDER, dataset, "semilep")
    ls = list(filter(lambda x : x.endswith(".parquet"), os.listdir(dataset_folder)))
    tables = []
    for file in ls[1:]:
        print(f"Reading {file}")
        tables.append(standardize_schema(pq.read_table(os.path.join(dataset_folder,file))))

    # Now you can write them together
    combined_table = pa.concat_tables(tables)
    output_file = os.path.join(OUTPUT_FOLDER, dataset, 'semilep', 'combined_output.parquet')
    output_folder = os.path.dirname(output_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pq.write_table(combined_table, output_file)

