apptainer shell \
  -B /afs \
  -B /cvmfs/cms.cern.ch \
  -B /tmp \
  -B /eos/cms \
  -B /eos/user/s/sedurgut \
  -B /eos/user/s/sedurgut/ttHbb/ttHbb-fully-hadronic \
  -B /etc/sysconfig/ngbauth-submit \
  -B ${XDG_RUNTIME_DIR} \
  -B /cvmfs/cms-griddata.cern.ch/cat/metadata \
  -B /cvmfs/cms-griddata.cern.ch \
  -B /cvmfs \
  --nv /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:latest

