import os
import argparse

import htcondor

from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument("-of", "--options_file", type=str, default=None,
                    help="JSON file with option overloads.", required=True)
parser.add_argument("-l", "--log_dir", type=str, default=None,
                    help="Output directory for the checkpoints and tensorboard logs. Default to current directory.", required=True)
parser.add_argument("-cf", "--checkpoint", type=str, default=None,
                    help="Optional checkpoint to load the training state from. "
                         "Fully restores model weights and optimizer state.")
parser.add_argument('--dry', action="store_true")
parser.add_argument('--interactive', action="store_true")
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--ncpu', type=int, default=3)
parser.add_argument("--good-gpus", action="store_true")
parser.add_argument("--args", nargs="+", type=str, help="additional args")
args = parser.parse_args()

interactive = args.interactive

col = htcondor.Collector()
credd = htcondor.Credd()
credd.add_user_cred(htcondor.CredTypes.Kerberos, None)

cfg = OmegaConf.load(args.cfg)
basedir = cfg['path']
model = cfg['model']
job_flavour = cfg['job_flavour']
ngpu = cfg['ngpu']
ncpu = cfg['ncpu']

# Override defaults by command line arguments
if args.ngpu != parser.get_default("ngpu"):
    ngpu = args.ngpu
if args.ncpu != parser.get_default("ncpu"):
    ncpu = args.ncpu

print("Initializing job submission script...", end="\n\n")
sub = htcondor.Submit()

if interactive:
    sub['InteractiveJob'] = True

if model in ["jet_assignment", "classification", "classification_balance_weights", "test_validation_loss", "test_mdmm_loss"]:
    sub['Executable'] = f"{basedir}/jobs/{model}.sh"
    sub['arguments'] = f"{args.options_file} {args.log_dir}"
    sub['Output'] = f"{basedir}/jobs/output/{model}-$(ClusterId).$(ProcId).out"
    sub['Error'] = f"{basedir}/jobs/error/{model}-$(ClusterId).$(ProcId).err"
    sub['Log'] = f"{basedir}/jobs/log/{model}-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:latest"'
    sub['+JobFlavour'] = f'"{job_flavour}"'
else:
    raise ValueError(f"Model {model} not implemented")

# Load checkpoint
if args.checkpoint:
    sub['arguments'] += f" {args.checkpoint}"

# GPU and CPU requirements
sub['request_cpus'] = f"{args.ncpu}"
sub['request_gpus'] = f"{args.ngpu}"

if args.good_gpus:
    sub['requirements'] = 'regexp("A100", TARGET.GPUs_DeviceName) || regexp("V100", TARGET.GPUs_DeviceName)'

print("Submission parameters:")
print(sub, end="\n\n")

print("Creating log folders...")
for folder in ["Output", "Error", "Log"]:
    os.makedirs(os.path.dirname(sub[folder]), exist_ok=True)
 
if not args.dry:
    print("Starting Condor scheduler...")
    client = htcondor.Schedd()
    result = client.submit(sub, count=1)

    print(f"Submitted {result.num_procs()} job(s) to {result.cluster()}")
