import awkward as ak

def mass_t1(jets):
    return (jets[:,0] + jets[:,1] + jets[:,2]).m

def pt_t1(jets):
    return (jets[:,0] + jets[:,1] + jets[:,2]).pt

def eta_t1(jets):
    return (jets[:,0] + jets[:,1] + jets[:,2]).eta

def phi_t1(jets):
    return (jets[:,0] + jets[:,1] + jets[:,2]).phi

def mass_h(jets):
    return (jets[:,0] + jets[:,1]).m

def pt_h(jets):
    return (jets[:,0] + jets[:,1]).pt

def eta_h(jets):
    return (jets[:,0] + jets[:,1]).eta

def phi_h(jets):
    return (jets[:,0] + jets[:,1]).phi

def mass_t2(jets, lep, met):
    return ak.flatten((lep + met + jets[:,0]).m)

def pt_t2(jets, lep, met):
    return ak.flatten((lep + met + jets[:,0]).pt)
    
def eta_t2(jets, lep, met):
    return ak.flatten((lep + met + jets[:,0]).eta)
    
def phi_t2(jets, lep, met):
    return ak.flatten((lep + met + jets[:,0]).phi)
