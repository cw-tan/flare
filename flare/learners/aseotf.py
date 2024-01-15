import numpy as np
import ase
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read
from typing import Optional, Callable, List
from flare.bffs.sgp._C_flare import SparseGP, B2, Structure  # , NormalizedDotProduct
from flare.bffs.sgp.sparse_gp import optimize_hyperparameters
from flare.bffs.sgp.calculator import sort_variances
import logging
import os
import time



eVperA3_to_GPa = 160.21766208

formatter = logging.Formatter('%(asctime)s: %(message)s')


def setup_logger(log_file, level=logging.INFO):
    """
    To set up loggers
    """
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


def ase2flare(ase_stress):
    """
    Converts the ASE convention of storing stresses (Voigt) to the
    FLARE convention of storing stresses.
    """
    return -np.array([ase_stress[0],
                      ase_stress[5],
                      ase_stress[4],
                      ase_stress[1],
                      ase_stress[3],
                      ase_stress[2]])


def flare2ase(flare_stress):
    """
    Converts the FLARE convention of storing stresses to the
    more convenional ASE convention (Voigt).
    """
    return -np.array([flare_stress[0],
                      flare_stress[3],
                      flare_stress[5],
                      flare_stress[4],
                      flare_stress[2],
                      flare_stress[1]])


class FlareOTF(Calculator):
    """
    FLARE on-the-fly ASE Calculator

    Several Bayesian force field hyperparameters are fixed for a more seamless
    user experience, with some user-given inputs allowed.
        * Descriptors
            Fixed     : B2 ACE descriptors with a quadratic cutoff and Chebyshev
                        radial basis
            User-given: radial and angular fiedlity parameters (nmax, lmax)
        * Sparse GP Kernel
            Fixed     : normalized dot product kernel
            User-given: initial signal variance and power of kernel
        * Active Learning Mode
            Fixed     : local cluster variances are used to determine whether
                        DFT is called or not (instead of energy, forces or stress
                        variances) [MAY BE CHANGED IN THE FUTURE]

    Init Args:
        sgp_params (tuple or list)    : Parameters of a SparseGP object in the format
                                        (kernel_variance, kernel_power, energy_noise,
                                        force_noise, stress_noise).
        desc_params (tuple or list)   : Parameters for the B2 descriptor (nmax, lmax).
        rcut (list of list)           : Cut-off radii in a symmetric matrix whose order
                                        corresponds to the number of atomic species.
        species (list)                : List of atomic numbers indicating the species
                                        considered in the FLARE model.
        dftcalc (ASE Calculator)      : An ASE calculator object that does DFT.
        energy_correction (list)      : Per-type corrections to the DFT potential energy.
        train_EFS (list/tuple of bool): Whether to train on (energies, forces, stresses).
                                        Default is (True, True, True).
        dft_call_threshold (float)    : DFT is called when the uncertainty of any atomic
                                        environment exceeds this argument. Default is 0.0005.
        dft_add_threshold (float)     : Atomic environments whose uncertainties exceed this
                                        argument are added to the SGP model. This argument
                                        is recommended to be 5 to 10 times smaller than
                                        dft_call_threshold to pre-emptively sample uncertain
                                        environments. This argument must be smaller than
                                        dft_call_threshold, otherwise DFT might be called but
                                        no training data is added to the model. Default is 0.0001.
        dft_xyz_fname (string)        : Filename for DFT frames in .xyz format to be saved.
                                        Should contain '*' (replaced with the current step).
                                        Default argument (None) will not save the frames.
        std_xyz_fname (string)        : Filename for DFT frames in .xyz format to be saved.
                                        Should contain '*' (replaced with the current step).
                                        Default argument (None) will not save the frames.
        hyperparameter_optimization   : Boolean function that determines whether to run
                                        hyperparameter optimization, as a function of this
                                        OTF_ase_calc and an ASE Atoms object, i.e.
                                        func(OTF_ase_calc, atoms) -> bool.
        opt_bounds (list of floats)   : Bounds for the hyperparameter optimization.
        opt_method (string)           : Algorithm for the hyperparameter optimization. Only
                                        accepts ``L-BFGS-B``, ``BFGS`` and ``nelder-mead``.
        opt_itrations (int)           : Max number of iterations for the hyperparameter optimization.
        always_DFT (bool)             : Whether to always do DFT or not (useful for generating
                                        benchmarking data for hyperparameter tuning).
        never_learn (bool)            : Whether the SGP stops learning or not.
        path (str)                    : Path where the FLARE ASEOTF output files are generated in.
                                        Default is the path to where this object is instantiated.
    """

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self,
                 sgp_params: tuple or list,
                 desc_params: tuple or list,
                 rcut: list,
                 species: tuple or list,
                 dftcalc: Calculator,
                 energy_correction: Optional[list] = None,
                 train_EFS: Optional[list] = [True, True, True],
                 dft_call_threshold: Optional[float] = 0.0005,
                 dft_add_threshold: Optional[float] = 0.0001,
                 dft_xyz_fname: Optional[str] = None,
                 std_xyz_fname: Optional[str] = None,
                 hyperparameter_optimization: Callable[([object, object], bool)]
                                              = lambda flare_otf_calc, atoms: False,
                 opt_bounds: Optional[List[float]] = None,
                 opt_method: Optional[str] = 'L-BFGS-B',
                 opt_iterations: Optional[int] = 200,
                 always_DFT: Optional[bool] = False,
                 never_learn: Optional[bool] = False,
                 path: Optional[str] = None,
                 wandb: object = None,
                 ) -> object:

        # initialize ASE calculator attributes
        Calculator.__init__(self)

        # SGP parameters
        assert len(sgp_params) == 4, 'arg sgp_params should have 4 elements'
        # we expect sgp_params = (kernel, sigma_e, sigma_f, sigma_s)
        # seg fault occurs if we instantiate the kernel here
        # kernel = NormalizedDotProduct(sgp_params[0], sgp_params[1])
        self.sparse_gp = SparseGP([sgp_params[0]], sgp_params[1], sgp_params[2], sgp_params[3])
        self.sparse_gp.Kuu_jitter = 1e-8  # trick to ensure Kuu is positive semi-definite for Cholesky

        # three basic parameters that are changed for each calculation
        # i.e. atoms is updated at each calculate call, which would update coded species (convenience variable)
        # and the structure object (which is the one that interacts with the SGP object)
        self.atoms = None
        self.coded_species = None
        self.structure = None

        # force field parameters
        self.species = species
        self.rcut = np.array(rcut)
        assert self.rcut.shape[0] == self.rcut.shape[1], 'arg rcut should be a square matrix'
        assert np.allclose(self.rcut, self.rcut.T), 'arg rcut should be a symmetric matrix'
        assert self.rcut.shape[0] == len(species), 'order of matrix arg rcut should be len(species)'
        assert len(desc_params) == 2, 'arg desc_params should have 4 elements'
        self.descriptors = np.atleast_1d(B2('chebyshev', 'quadratic', [0.0, self.rcut.max()], [],
                                            [len(species), desc_params[0], desc_params[1]], self.rcut))
        if energy_correction is None:
            self.energy_correction = np.zeros(len(species))
        else:
            self.energy_correction = np.atleast_1d(energy_correction)
            assert len(self.energy_correction) == len(species), \
                'require len(species) = len(enrgy_corrections)'
        assert isinstance(dftcalc, Calculator), 'arg dftcalc must be an ASE Calculator object'
        self.dftcalc = dftcalc

        # training parameters
        assert dft_call_threshold > dft_add_threshold, \
            'require dft_call_threshold > dft_add_threshold'
        self.dft_call_threshold = dft_call_threshold
        self.dft_add_threshold = dft_add_threshold
        assert isinstance(train_EFS, tuple) or isinstance(train_EFS, list), \
            'arg train_EFS should be a tuple or list'
        assert len(train_EFS) == 3, 'arg train_EFS should have 3 elements'
        assert np.all([isinstance(train_EFS[i], bool) for i in range(3)]), \
            'arg train_EFS should have all boolean elements'
        self.train_EFS = train_EFS
        self.always_DFT = always_DFT
        self.never_learn = never_learn

        # recording frames
        if dft_xyz_fname is not None:
            assert isinstance(dft_xyz_fname, str), 'arg dft_xyz_fname must be a string'
            assert '*' in dft_xyz_fname, 'arg dft_xyz_fname should contain a * symbol'
        self.dft_xyz_fname = dft_xyz_fname
        if std_xyz_fname is not None:
            assert isinstance(std_xyz_fname, str), 'arg std_xyz_fname must be a string'
            assert '*' in std_xyz_fname, 'arg std_xyz_fname should contain a * symbol'
        self.std_xyz_fname = std_xyz_fname

        # hyperparameter optimization parameters
        assert isinstance(opt_method, str), 'arg opt_method must be a string'
        assert opt_method in ['L-BFGS-B', 'BFGS', 'nelder-mead']
        self.hyperparameter_optimization = hyperparameter_optimization
        self.opt_bounds = opt_bounds
        self.opt_method = opt_method
        self.opt_iterations = opt_iterations

        # recording purposes
        if path is not None:
            if path.endswith('/'):
                self.path = path[:-1]
            else:
                self.path = path
        else:
            self.path = os.getcwd()
        self.calls = 0  # total calls to the calculator
        self.dn_atenvs_since_hypopt = 0  # number of atomic environments added since last hyerparameter optimization
        self.hyp_opts = 0  # number of hyperparameter optimizations done
        self.dft_calls = 0
        self.last_dft_call = -100

        # for recording purposes
        self.logger = setup_logger(self.path + '/flare_aseotf.log')
        self.wandb = wandb

        self.time_dft = 0.0
        self.time_hyp_opt = 0.0
        self.time_training = 0.0
        self.time_predict_uncertainties = 0.0
        self.time_prediction = 0.0
        self.time_ase = 0.0

    def calculate(self, atoms, properties=None, system_changes=all_changes):
        """
        Calculate properties including: energy, local energies, forces,
        stress, uncertainties.
        """
        try:
            self.logger.info('-------------------- Call {} --------------------'.format(self.calls))
            self.__reset_structure(atoms)  # sets up the SGP structure object (among other things)
            natoms = len(atoms.numbers)
            if self.dft_calls == 0 and self.calls == 0:
                pe, F, stress = self.__run_dft()
                self.__update_sgp(init=True, atoms_to_add=int(natoms / 2))  # randomly choose half
            else:
                # get local cluster uncertainties
                t0 = time.time()
                sigma = self.sparse_gp.hyperparameters[0]
                variances = sort_variances(self.structure, self.sparse_gp.compute_cluster_uncertainties(self.structure)[0])
                self.time_predict_uncertainties += time.time() - t0
                stds = np.sqrt(np.abs(variances)) / sigma
                call_dft = np.any(stds > self.dft_call_threshold)
                atoms2add = np.arange(natoms)[stds > self.dft_add_threshold]

                # log cluster uncertainty (and save frames) 
                self.logger.info('Max cluster uncertainty      : {:.16f}'.format(stds.max()))
                if self.std_xyz_fname is not None:
                    frame = atoms.copy()
                    frame.set_array('charges', stds)
                    ase.io.write(self.std_xyz_fname.replace('*', str(self.calls)), frame, format='extxyz')

                if self.never_learn or not call_dft:  # just get mean, uncertainties not necessary 
                    # update ASE calculator results (energy, forces, stress) with SGP
                    predE, predF, predS = self.__predict_from_sgp(mode='mean')
                    self.results["energy"] = predE
                    self.results["forces"] = predF
                    self.results["stress"] = predS
                else: 
                    # call DFT and update SGP model
                    pe, forces, stress = self.__run_dft()

                    if not self.always_DFT:
                        predE, predF, predS = self.__predict_from_sgp(mode='dtc')
                        # log data based on DFT calculation
                        errF = np.abs(forces - predF)
                        errS = np.abs(stress - predS)
                        self.logger.info('DFT energy error [meV/atom]  : {:^14.8f}'.format(1000 * np.abs(pe - predE) / natoms))
                        self.logger.info('Max & Mean DFT force error [meV/Å]  : {:^14.8f} {:^14.8f}'
                                         .format(1000 * errF.max(), 1000 * np.mean(errF)))
                        self.logger.info('Max & Mean DFT stress error [GPa]   : {:^14.8f} {:^14.8f}'
                                         .format(eVperA3_to_GPa * errS.max(), eVperA3_to_GPa * np.mean(errS)))
                        if self.wandb is not None:  # wandb logging
                            wandb_log = {'max_cluster_uncertainty': stds.max()}
                            wandb_log['max_force_error'] = 1000 * errF.max()
                            wandb_log['mean_force_error'] = 1000 * np.mean(errF)
                            wandb_log['log_rel_force_error'] = self.wandb.Histogram(np.log10(errF / np.abs(forces)).ravel())
                            wandb_log['max_stress_error'] = eVperA3_to_GPa * errS.max()
                            wandb_log['mean_stress_error'] = eVperA3_to_GPa * np.mean(errS)
                            wandb_log['log_rel_stress_error'] = self.wandb.Histogram(np.log10(errS / np.abs(stress)).ravel())
                            wandb_log['aseotf_call'] = self.calls
                            self.wandb.log(wandb_log, step=self.dft_calls)  # only log when DFT called

                        # update SGP and potentially optimize SGP hyperparameters
                        self.__update_sgp(init=False, atoms_to_add=atoms2add)
                        if self.hyperparameter_optimization(self, self.atoms):
                            t0 = time.time()
                            optimize_hyperparameters(self.sparse_gp,
                                                     bounds=self.opt_bounds,
                                                     method=self.opt_method,
                                                     max_iterations=self.opt_iterations,
                                                     logger=self.logger)
                            self.time_hyp_opt += time.time() - t0
                            self.hyp_opts += 1
                            self.dn_atenvs_since_hypopt = 0

                        if self.wandb is not None:  # wandb logging for hyperparameters
                            wandb_log = {'kernel_outputscale': np.abs(self.sparse_gp.hyperparameters[0])}
                            wandb_log['energy_noise'] = np.abs(self.sparse_gp.hyperparameters[1])
                            wandb_log['force_noise'] = 1000 * np.abs(self.sparse_gp.hyperparameters[2])
                            wandb_log['stress_noise'] = eVperA3_to_GPa * np.abs(self.sparse_gp.hyperparameters[3])
                            self.wandb.log(wandb_log, step=self.dft_calls)  # only log when DFT called

            self.calls += 1
            self.record_state(self.path + '/flare_aseotf.state')

        except Exception as err:
            try:
                self.logger.exception('ASEOTF error')
                raise err
            finally:
                err = None
                del err

    def offline_train(self, dft_xyz_fname):
        """
        Performs offline training.
        Args:
            dft_xyz_fname (string): Filename for DFT frames in .xyz format to be saved.
                                    Should contain '*' (replaced with the 'current step').
                                    Should be reachable from the location where this
                                    function is called.
        """
        t_offline_start = time.time()

        # get the frame names in order
        # split potential path and frame name convention
        frame_fname = dft_xyz_fname.split('/')[-1]
        path = dft_xyz_fname.replace(frame_fname, '')
        before, after = frame_fname.split('*')
        frames = []  # collect frames from directory
        for dir_entry in sorted(os.listdir(path)):
            if dir_entry.startswith(before) and dir_entry.endswith(after):
                frames.append(dir_entry)
        # rearrange frames
        frame_nums = [int(frame.replace(before, '').replace(after, '')) for frame in frames]
        paths2frames = [path + x for _, x in sorted(zip(frame_nums, frames))]

        # start offline training
        self.logger.info('-------------------- Start of Offline Training (Call 0) --------------------')
        for j, path2frame in enumerate(paths2frames):
            self.logger.info('-------------------- Frame {} --------------------'.format(sorted(frame_nums)[j]))
            # read xyz file and extract structure, energy, forces and stress
            frame = read(path2frame)
            natoms = len(frame.numbers)
            pe = frame.get_potential_energy()
            if not isinstance(pe, float):
                pe = float(pe.split(' ')[0])  # dirty workaround
            forces = frame.get_forces()
            stress = frame.get_stress()
            self.__reset_structure(frame)  # sets up the SGP structure object (among other things)
            if j == 0:
                self.__update_structure_efs(pe, forces, stress)
                self.__update_sgp(init=True, atoms_to_add=int(natoms / 2))
            else:
                # get uncertainties
                t0 = time.time()
                sigma = self.sparse_gp.hyperparameters[0]
                variances = sort_variances(self.structure, self.sparse_gp.compute_cluster_uncertainties(self.structure)[0])
                self.time_predict_uncertainties += time.time() - t0
                stds = np.sqrt(np.abs(variances)) / sigma
                add_envs = np.any(stds > self.dft_call_threshold)
                self.logger.info('Max cluster uncertainty      : {:.16f}'.format(stds.max()))

                predE, predF, predS = self.__predict_from_sgp(mode='dtc') 
                self.__update_structure_efs(pe, forces, stress)

                # log data based on DFT calculation
                errF = np.abs(forces - predF)
                errS = np.abs(stress - predS)
                self.logger.info('DFT energy error [meV/atom]  : {:^14.8f}'.format(1000 * np.abs(pe - predE) / natoms))
                self.logger.info('Max & Mean DFT force error [meV/Å]  : {:^14.8f} {:^14.8f}'
                                 .format(1000 * errF.max(), 1000 * np.mean(errF)))
                self.logger.info('Max & Mean DFT stress error [GPa]   : {:^14.8f} {:^14.8f}'
                                 .format(eVperA3_to_GPa * errS.max(), eVperA3_to_GPa * np.mean(errS)))
                if self.wandb is not None:  # wandb logging
                    wandb_log = {'max_cluster_uncertainty': stds.max()}
                    wandb_log['max_force_error'] = 1000 * errF.max()
                    wandb_log['mean_force_error'] = 1000 * np.mean(errF)
                    wandb_log['log_rel_force_error'] = self.wandb.Histogram(np.log10(errF / np.abs(forces)).ravel())
                    wandb_log['max_stress_error'] = eVperA3_to_GPa * errS.max()
                    wandb_log['mean_stress_error'] = eVperA3_to_GPa * np.mean(errS)
                    wandb_log['log_rel_stress_error'] = self.wandb.Histogram(np.log10(errS / np.abs(stress)).ravel())
                    wandb_log['aseotf_call'] = self.calls
                    self.wandb.log(wandb_log, step=self.dft_calls)  # only log when DFT called
                if add_envs:
                    self.dft_calls += 1  # to reproduce hyperparameter optimization calls
                    # update SGP and potentially optimize SGP hyperparameters
                    atoms2add = np.arange(natoms)[stds > self.dft_add_threshold]
                    self.__update_sgp(init=False, atoms_to_add=atoms2add)
                    if self.hyperparameter_optimization(self, self.atoms):
                        t0 = time.time()
                        optimize_hyperparameters(self.sparse_gp,
                                                 bounds=self.opt_bounds,
                                                 method=self.opt_method,
                                                 max_iterations=self.opt_iterations,
                                                 logger=self.logger)
                        self.time_hyp_opt += time.time() - t0
                        self.hyp_opts += 1
                        self.dn_atenvs_since_hypopt = 0

                if self.wandb is not None:  # wandb logging for hyperparameters
                    wandb_log = {'kernel_outputscale': np.abs(self.sparse_gp.hyperparameters[0])}
                    wandb_log['energy_noise'] = np.abs(self.sparse_gp.hyperparameters[1])
                    wandb_log['force_noise'] = 1000 * np.abs(self.sparse_gp.hyperparameters[2])
                    wandb_log['stress_noise'] = eVperA3_to_GPa * np.abs(self.sparse_gp.hyperparameters[3])
                    self.wandb.log(wandb_log, step=self.dft_calls)

            self.calls += 1

        offline_training_time = time.time() - t_offline_start
        self.logger.info('Time spent offline training: {} s'.format(offline_training_time))
        self.logger.info('-------------------- End of Offline Training --------------------')
        self.dft_calls = 0
        self.calls = 1  # mark to avoid initialization DFT step when deployed
        self.record_state(self.path + '/flare_aseotf.state')

    def __run_dft(self):
        self.dft_calls += 1
        self.logger.info('Calling DFT... (DFT call #{})'.format(self.dft_calls))

        # copy Atoms object just in case ... and set DFT calculator
        frame = self.atoms.copy()
        frame.calc = self.dftcalc

        # get DFT energy, forces, stress
        t0 = time.time()  # record DFT time
        pe = frame.get_potential_energy()
        pe -= np.sum(self.energy_correction[self.coded_species])
        forces = frame.get_forces()
        stress = frame.get_stress(voigt=True)
        self.time_dft += time.time() - t0

        # record DFT frame
        if self.dft_xyz_fname is not None:
            ase.io.write(self.dft_xyz_fname.replace("*", str(self.calls)),
                         frame, format="extxyz")

        # update structure object with DFT outputs for training
        self.__update_structure_efs(pe, forces, stress)

        # update ASE calculator results (energy, forces, stress) with DFT
        self.results['energy'] = pe
        self.results['forces'] = forces
        self.results['stress'] = stress

        self.last_dft_call = self.calls
        return pe, forces, stress

    def __reset_structure(self, atoms):
        """
        Internal function to set SGP Structure object, self.structure.
        """
        self.atoms = atoms

        # maps the atoms in the ASE Atoms object to their indices in the SGP object
        assert set(np.unique(atoms.numbers)).issubset(set(self.species)), \
            'argument species during init inconsistent with the given configuration'
        species_map = {self.species[i]: i for i in range(len(self.species))}
        coded_species = []
        for spec in atoms.numbers:
            coded_species.append(species_map[spec])
        self.coded_species = coded_species  # convenience variable reused in self.__run_DFT()

        # reset self.structure
        self.structure = Structure(atoms.cell, self.coded_species, atoms.positions,
                                   self.rcut.max(), self.descriptors)

    def __update_structure_efs(self, e, f, s):
        if self.train_EFS[0]:
            self.structure.energy = np.array([e])
        if self.train_EFS[1]:
            self.structure.forces = f.reshape(-1)
        if self.train_EFS[2]:
            self.structure.stresses = ase2flare(s)

    def __update_sgp(self, init, atoms_to_add):
        """
        Internal function to update SGP with structure and its uncertain atomic environments,
        along with the corresponding energy, forces and stress. i.e. this trains the model.

        If used for initialization (init=True), atoms_to_add is the number of atoms to add.
        Otherwise, atoms_to_add is a list of atom indices to add.
        """
        t0 = time.time()
        self.sparse_gp.add_training_structure(self.structure)
        if init:
            self.sparse_gp.add_random_environments(self.structure, [atoms_to_add])
            self.logger.info('Added {} random atomic environments'.format(atoms_to_add))
            self.dn_atenvs_since_hypopt += atoms_to_add
        else:
            self.sparse_gp.add_specific_environments(self.structure, atoms_to_add)
            self.logger.info('Added {} atomic environments'.format(len(atoms_to_add)))
            self.dn_atenvs_since_hypopt += len(atoms_to_add)
        self.sparse_gp.update_matrices_QR()
        self.time_training += time.time() - t0

    def __predict_from_sgp(self, mode='mean'):
        """
        mode (str): 'mean' (only mean) or 'dtc' (mean & DTC variance)
        """
        # get SGP prediction
        assert mode == 'mean' or mode == 'dtc'
        t0 = time.time()
        if mode == 'mean':
            self.sparse_gp.predict_mean(self.structure)
        elif mode == 'dtc':
            self.sparse_gp.predict_DTC(self.structure)
        self.time_prediction += time.time() - t0
         
        # collect mean predictions
        predE = self.structure.mean_efs[0]
        predF = self.structure.mean_efs[1:-6].reshape((-1, 3)).copy()
        predS = flare2ase(self.structure.mean_efs[-6:].copy())
        natoms = predF.shape[0]

        if mode == 'dtc':  # log uncertainties
            # get DTC SGP variance (epistemic uncertainty)
            Evar = np.abs(self.structure.variance_efs[0])
            Fvar = np.abs(self.structure.variance_efs[1:-6]).reshape((-1, 3))
            Svar = np.abs(self.structure.variance_efs[-6:])
            # combine with noise hyperparameter uncertainty
            E_noise, F_noise, S_noise = self.sparse_gp.hyperparameters[1:4]
            E_uncertainty = np.sqrt((Evar + E_noise * E_noise) / natoms) * 1000
            F_uncertainty = np.sqrt(Fvar + F_noise * F_noise) * 1000
            S_uncertainty = np.sqrt(Svar + S_noise * S_noise).max() * eVperA3_to_GPa
            # log uncertainties for energy, forces and stress
            # format: total uncertainty, epistemic uncertainty, model noise
            self.logger.info('Energy uncertainty [meV/atom]: {:^14.8f}  {:^14.8f}  {:^14.8f}'
                             .format(E_uncertainty, np.sqrt(Evar / natoms) * 1000, np.abs(E_noise) * 1000 / np.sqrt(natoms)))
            self.logger.info('Max force uncertainty [meV/Å]: {:^14.8f}  {:^14.8f}  {:^14.8f}'
                             .format(F_uncertainty.max(), np.sqrt(Fvar).max() * 1000, np.abs(F_noise) * 1000))
            self.logger.info('Max stress uncertainty [GPa] : {:^14.8f}  {:^14.8f}  {:^14.8f}'
                             .format(S_uncertainty, np.sqrt(Svar).max() * eVperA3_to_GPa, np.abs(S_noise) * eVperA3_to_GPa))
        return predE, predF, predS

    def record_state(self, fname):

        with open(fname, 'w') as f:
            f.write('FLARE ASE-OTF STATUS\n')
            f.write('Total Calls         : {}\n'.format(self.calls))
            f.write('DFT Calls           : {}\n'.format(self.dft_calls))
            f.write('DFT Call Percentage : {:.2f}%\n\n'.format(self.dft_calls / self.calls * 100))
            f.write('SPARSE GP STATUS\n')
            if self.train_EFS[1]:  # so that the force labels are meaningful
                f.write('Full atomic envs.   : {}\n'.format(int(self.sparse_gp.n_force_labels / 3)))
            f.write('Sparse atomic envs. : {}\n'.format(self.sparse_gp.n_sparse))
            f.write('Total labels        : {}\n'.format(self.sparse_gp.n_labels))
            f.write('Energy labels       : {}\n'.format(self.sparse_gp.n_energy_labels))
            f.write('Force labels        : {}\n'.format(self.sparse_gp.n_force_labels))
            f.write('Stress labels       : {}\n\n'.format(self.sparse_gp.n_stress_labels))
            # strangely, self.sparse_gp.n_strucs doesn't work (possible TODO)
            # though, energy_labels gives the same information (I think)

            f.write('HYPERPARAMETERS\n')
            f.write('Kernel Noise        : {:.10f}\n'.format(np.abs(self.sparse_gp.hyperparameters[0])))
            f.write('Energy Noise [eV]   : {:.10f}\n'.format(np.abs(self.sparse_gp.hyperparameters[1])))
            f.write('Force Noise [meV/Å] : {:.10f}\n'.format(1000 * np.abs(self.sparse_gp.hyperparameters[2])))
            f.write('Stress Noise [GPa]  : {:.10f}\n'.format(eVperA3_to_GPa * np.abs(self.sparse_gp.hyperparameters[3])))
            f.write('Times optimized     : {}\n'.format(self.hyp_opts))
