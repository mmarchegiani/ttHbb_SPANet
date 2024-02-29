import htcondor
import sys
import argparse
import os
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--version', type=str, required=True)
parser.add_argument('--path-configFile', type=str, required=True)
parser.add_argument('--dry', action="store_true")
parser.add_argument('--interactive', action="store_true")
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--ncpu', type=int, default=3)
parser.add_argument("--good-gpus", action="store_true")
parser.add_argument("--args", nargs="+", type=str, help="additional args")
args = parser.parse_args()

model = args.model
version = args.version
dry = args.dry
interactive = args.interactive
path_to_conf = args.path_configFile

col = htcondor.Collector()
credd = htcondor.Credd()
credd.add_user_cred(htcondor.CredTypes.Kerberos, None)

# Read config file in 'conf'
with open(path_to_conf) as f:
    try:
        conf = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

print("Path to root & outputDir: \n", conf)

basedir = conf['path_to_root']
outputDir = conf['path_to_outputDir']

sub = htcondor.Submit()

if interactive:
    sub['InteractiveJob'] = True

if model == "huber_mmd":
    sub['Executable'] = f"{basedir}/jobs/script_condor_pretraining_huber_mmd.sh"
    sub['Error'] = f"{basedir}/jobs/error/huber-mmd-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/huber-mmd-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/huber-mmd-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"{basedir} configs/pretraining_huber_mmd/pretraining_huber_mmd_v{version}.yaml {outputDir}/flow_pretraining_huber_mmd"
else:
    raise ValueError(f"Model {model} not implemented")

# General
sub['request_cpus'] = f"{args.ncpu}"
sub['request_gpus'] = f"{args.ngpu}"
if args.good_gpus:
    sub['requirements'] = 'regexp("A100", TARGET.CUDADeviceName) || regexp("V100", TARGET.CUDADeviceName)'

print(f"{basedir}/{sub['arguments'].split()[1]}")

if model != "flow_evaluation_labframe" and not os.path.exists(f"{basedir}/{sub['arguments'].split()[1]}"):
    print("Missing configuration file! The jobs has not been submitted")
    exit(1)
    
print(sub)
if not dry:
    schedd = htcondor.Schedd()
    with schedd.transaction() as txn:
        cluster_id = sub.queue(txn)

    print(f"Submitted to {cluster_id:=}")
