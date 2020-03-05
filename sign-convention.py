#!/usr/bin/env python3

"""
Example script to show how to set up the HMC driver for production including thermalization.
"""

# All output should go through a logger for better control and consistency.
from logging import getLogger

# Import base functionality of Isle.
import isle
# Import drivers (not included in above).
import isle.drivers
import isle.meas

from pathlib import Path

from itertools import product
from numpy import s_
import numpy as np
import h5py as h5


def DISC(discretization):
    if discretization == isle.action.HFAHopping.DIA:
        return "diagonal"
    elif discretization == isle.action.HFAHopping.EXP:
        return "exponential"
    return "???"

def OUTFILE(lattice, nt, discretization):
    return Path(f"{lattice}.nt{nt}.{discretization}.h5")

def PARAMS(discretization):

    return isle.util.parameters(
        beta=1.75,      # inverse temperature
        U=5,            # on-site coupling
        mu=0,           # chemical potential
        sigmaKappa=-1,  # prefactor of kappa for holes / spin down
                    # (+1 only allowed for bipartite lattices)

        # Those three control which implementation of the action gets used.
        # The values given here are the defaults.
        # See documentation in docs/algorithm.
        hopping={"diagonal": isle.action.HFAHopping.DIA, "exponential": isle.action.HFAHopping.EXP}[discretization],
        basis=isle.action.HFABasis.PARTICLE_HOLE,
        variant=isle.action.HFAVariant.ONE
    )

def makeAction(lat, params):
    # Import everything this function needs so it is self-contained.
    import isle
    import isle.action

    return isle.action.HubbardGaugeAction(params.tilde("U", lat)) \
        + isle.action.makeHubbardFermiAction(lat,
                                             params.beta,
                                             params.tilde("mu", lat),
                                             params.sigmaKappa,
                                             params.hopping,
                                             params.basis,
                                             params.variant)

def hmc(lat, params, file, log):

    LATTICE=lat.name
    nt=lat.nt()
    discretization=DISC(params.hopping)

    rng = isle.random.NumpyRNG(1075)

    hmcState = isle.drivers.hmc.newRun(lat, params, rng, makeAction,
                                       file, False)

    # Generate a random initial condition.
    phi = isle.Vector(rng.normal(0,
                             params.tilde("U", lat)**(1/2),
                             lat.lattSize())
                      +0j)

    log.info(f"Thermalizing {LATTICE} nt={nt} {discretization}")
    evolver = isle.evolver.LinearStepLeapfrog(hmcState.action, (1, 1), (20, 5), 99, rng)
    evStage = hmcState(phi, evolver, 1000, saveFreq=0, checkpointFreq=0)
    hmcState.resetIndex()

    log.info(f"Producing {LATTICE} nt={nt} {discretization}")
    evolver = isle.evolver.ConstStepLeapfrog(hmcState.action, 1, 5, rng)
    hmcState(evStage, evolver, 100000, saveFreq=10, checkpointFreq=100)


def measure(lat, params, file, log):

    LATTICE=lat.name
    nt=lat.nt()
    discretization=DISC(params.hopping)

    measState = isle.drivers.meas.init(file, file, True)
    params = measState.params

    muTilde = params.tilde("mu", lat)
    kappaTilde = params.tilde(measState.lattice.hopping(), lat.nt())

    if params.hopping == isle.action.HFAHopping.DIA:
        hfm = isle.HubbardFermiMatrixDia(kappaTilde, muTilde, params.sigmaKappa)
    else:
        hfm = isle.HubbardFermiMatrixExp(kappaTilde, muTilde, params.sigmaKappa)

    species =   (isle.Species.PARTICLE, isle.Species.HOLE)
    allToAll = {s: isle.meas.propagator.AllToAll(hfm, s) for s in species}
    _, transformation = np.linalg.eigh(isle.Matrix(hfm.kappaTilde()))

    measurements = [
        isle.meas.Logdet(hfm, "logdet"),
        isle.meas.TotalPhi("field"),
        isle.meas.CollectWeights("weights"),
        isle.meas.onePointFunctions(
                            allToAll[isle.Species.PARTICLE],
                            allToAll[isle.Species.HOLE],
                            "correlation_functions/one_point",
                            configSlice=s_[::],
                            transform=None
                            ),
        isle.meas.SingleParticleCorrelator(
                            allToAll[isle.Species.PARTICLE],
                            "correlation_functions/single_particle",
                            configSlice=s_[::],
                            projector=transformation
                            ),
        isle.meas.SingleParticleCorrelator(
                            allToAll[isle.Species.HOLE],
                            "correlation_functions/single_hole",
                            configSlice=s_[::],
                            projector=transformation
                            ),
    ]

    measState(measurements)


def bootstrap(lat, params, file, log):

    LATTICE=lat.name
    nt=lat.nt()
    discretization=DISC(params.hopping)
    log.info(f"Reading measurements for {LATTICE} nt={nt} {discretization}")
    with h5.File(file, 'r') as f:
        measurements = {
            "N_h": f['/correlation_functions/one_point/N_h'][()],
            "N_p": f['/correlation_functions/one_point/N_p'][()],
            # "a_adag": f['/correlation_functions/single_particle/correlators'][()],
            # "b_bdag": f['/correlation_functions/single_hole/correlators'][()],
            }

        measurements["rho"] = measurements["N_p"] - measurements["N_h"]
        measurements["N"]   = measurements["N_p"] + measurements["N_h"]
        S = f['/weights/actVal'][()]

    N_BS=100
    samples = measurements["N_h"].shape[0]
    resamples = np.random.randint(0, samples, [N_BS, samples])

    # Pretty sure I got the sign right...
    weights = np.exp(-1j*np.imag(S))

    for m in measurements:
        bootstrapped = [ np.einsum('i,i...->...', weights[sample], measurements[m][sample]) / np.sum(weights[sample]) for sample in resamples ]
        mean = np.mean(bootstrapped, axis=0)
        std  = np.std (bootstrapped,  axis=0)
        with h5.File(file, 'a') as f:
            try:
                del f[f"bootstrapped/{m}-mean"]
                del f[f"bootstrapped/{m}-std"]
            except:
                pass
            f[f"bootstrapped/{m}-mean"] = mean
            f[f"bootstrapped/{m}-std"] = std

        print(f"{m:5}  {np.real(mean)} {np.real(std)}")

def main():

    isle.initialize("default")
    log = getLogger("HMC")

    lattices=["triangle-asymmetric"]#, "triangle"]
    nts=[32]#[128]#[64]#[16]
    discretizations=["exponential", "diagonal"]

    for LATTICE, nt, discretization in product(lattices, nts, discretizations):

        log.info(f"{LATTICE} nt={nt} {discretization}")
        print(f"{LATTICE} nt={nt} {discretization}")

        lat = isle.LATTICES[LATTICE]
        lat.nt(nt)

        params = PARAMS(discretization)
        file = OUTFILE(LATTICE,nt,discretization)

        hmc(lat, params, file, log)
        measure(lat, params, file, log)
        bootstrap(lat, params, file, log)

    # with h5.File("summary.h5", 'a') as summary:
    #     summary[f"{LATTICE}/{discretization}/{nt}"] = np.arange(10)

if __name__ == "__main__":
    main()
