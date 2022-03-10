import os.path as op
import numpy as np
import nibabel as nib
import pandas as pd
import seaborn as sns
from nistats.first_level_model import FirstLevelModel, make_first_level_design_matrix, run_glm
from nistats.contrasts import _fixed_effect_contrast, compute_contrast
from nistats.hemodynamic_models import glover_hrf
from nistats.design_matrix import _cosine_drift
from nilearn import datasets
import matplotlib.pyplot as plt
from nilearn.plotting import view_img
from nilearn import plotting, signal, masking, image
from glob import glob
from nilearn import masking
from nideconv import ResponseFitter
from tqdm import tqdm_notebook

from patsy import DesignInfo
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from warnings import warn
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-8s] [%(levelname)-7.7s]  %(message)s",
    handlers=[logging.StreamHandler()]
)


class Dataset:
    """ Dataset class for easy fMRI analyses with fMRIPREP outputs. """
    def __init__(self, bids_dir, sub, n_jobs=1, log_level=logging.INFO):
        """ Initializes Analysis object.
        
        parameters
        ----------
        bids_dir : str
            Path to BIDS directory
        sub : str
            Subject-identifier (e.g., '01')
        n_jobs : int
            Number of jobs to use for computations
        """
        self.sub = sub
        self.bids_dir = op.join(op.abspath(bids_dir), f'sub-{sub}')
        self.n_jobs = n_jobs

        self.deriv_dir = op.join(bids_dir, 'derivatives')
        self.fp_dir = op.join(self.deriv_dir, 'fmriprep', f'sub-{sub}')
        self.fs_dir = op.join(self.deriv_dir, 'freesurfer', f'sub-{sub}')
        self.physio_dir = op.join(self.deriv_dir, 'physiology', f'sub-{sub}')
        self.logger = logging.getLogger('dataset')
        self.logger.setLevel(log_level)

    def create_taskset(self, task, space, acq=None, ses=False, hemi='R', reference_run=-1,
                       use_gm_mask=True, gm_threshold=0.5):
        """ Creates a 'taskset'. 
        
        Parameters
        ----------
        task : str
            Name of task
        space : str
            Name of space to analyze data in (e.g., 'T1w', 'fsnative')
        acq : str
            Name of acquisition parameter (in filenames, e.g., acq-Mm3)
        ses : bool
            Whether the taskset is split over multiple sessions
        hemi : str
            Either 'L' or 'R'; only used when space == 'fsnative'
        """
        base_str = f'sub-{self.sub}'
        if ses:
            base_str += f'_ses-*'
            this_fp_dir = op.join(self.fp_dir, 'ses-*')
            this_bids_dir = op.join(self.bids_dir, 'ses-*')
            this_physio_dir = op.join(self.physio_dir, 'ses-*')
        else:
            this_fp_dir = self.fp_dir
            this_bids_dir = self.bids_dir
            this_physio_dir = self.physio_dir

        base_str += f'_task-{task}'

        if acq is not None:
            base_mri_str = f'{base_str}_acq-{acq}'
        else:
            base_mri_str = base_str
        
        self.logger.info(f"Proceeding with base MRI string '{base_mri_str}'")
        
        conf_str = '_desc-confounds_regressors.tsv'
        all_confs = sorted(glob(op.join(this_fp_dir, 'func', f'{base_mri_str}*{conf_str}')))
        self.logger.info(f"Found {len(all_confs)} confound files.")
        
        if 'fs' in space:
            func_str = f'_space-{space}_hemi-{hemi}.func.gii'
        else:
            func_str = f'_space-{space}_desc-preproc_bold.nii.gz'
        
        all_funcs = sorted(glob(op.join(this_fp_dir, 'func', f'{base_mri_str}*{func_str}')))
        self.logger.info(f"Found {len(all_funcs)} functional file(s).")
        
        if 'fs' not in space:        
            mask_str = f'_space-{space}_desc-brain_mask.nii.gz'
            all_masks = sorted(glob(op.join(this_fp_dir, 'func', f'{base_mri_str}*{mask_str}')))
            self.logger.info(f"Found {len(all_masks)} functional mask(s).")

            tmp = nib.load(all_masks[reference_run])
            for i, mask in enumerate(all_masks):  # check if we need to resapmle
                if not np.all(nib.load(mask).affine == tmp.affine):
                    self.logger.warning(
                        f"WARNING: mask of run {i} has a different affine than "
                        f"the reference run {reference_run}! Going to resample ..."
                    )
                    all_masks[i] = image.resample_img(
                        mask,
                        target_affine=tmp.affine,
                        target_shape=tmp.shape,
                        interpolation='nearest'
                    )
                
            mask = masking.intersect_masks(all_masks, threshold=0.5)
            if use_gm_mask:
                if 'MNI' in space:
                    gm_mask = op.join(
                        self.fp_dir, 'anat',
                        f'sub-{self.sub}_space-{space}_label-GM_probseg.nii.gz'
                    )
                else:
                    gm_mask = op.join(
                        self.fp_dir, 'anat',
                        f'sub-{self.sub}_label-GM_probseg.nii.gz'
                    )
                
                self.logger.info(f"Adding GM mask: {gm_mask}")
                    
                gm_img = nib.load(gm_mask)
                gm_mask = nib.Nifti1Image((gm_img.get_data() > gm_threshold).astype(int), affine=gm_img.affine)
                
                if not np.all(gm_mask.affine == tmp.affine):
                    self.logger.info(
                        "Resampling GM mask because has a different affine than the functional masks."
                    )
                    gm_mask = image.resample_img(
                        gm_mask,
                        target_affine=tmp.affine,
                        target_shape=tmp.shape,
                        interpolation='nearest'
                    )

                mask = masking.intersect_masks([mask, gm_mask], threshold=1)
        else:
            mask = None  # no mask for surface files!
            gm_mask = None

        event_str = '_events.tsv'
        all_events = sorted(glob(op.join(this_bids_dir, 'func', f'{base_mri_str}*{event_str}')))
        self.logger.info(f"Found {len(all_events)} event file(s).")
        
        if len(all_funcs) != len(all_events) != len(all_confs):
            self.logger.warning(
                f"Found {len(all_funcs)} funcs but {len(all_events)} events and {len(all_confs)} confs!"
            )
        
        ricor_str = '_desc-retroicor_regressors.tsv'
        all_ricors = sorted(glob(op.join(this_physio_dir, 'physio', f'{base_mri_str}*{ricor_str}')))
        self.logger.info(f"Found {len(all_ricors)} RETROICOR file(s).")
        
        if len(all_ricors) != len(all_funcs):
            self.logger.warning(
                f"Found {len(all_funcs)} funcs but only {len(all_ricors)} RETROICOR files!"
            )
        
        data = dict(funcs=dict(), events=dict(), confs=dict(), ricors=dict(), mask=mask)
        for run, (func, event, conf) in enumerate(zip(all_funcs, all_events, all_confs)):
                
            for name, elem in zip(['funcs', 'events', 'confs'], [func, event, conf]):
                data[name][run] = elem
                
            tmp_base = op.basename(conf.split(conf_str)[0])
            for r in all_ricors:
                if tmp_base in r:
                    data['ricors'][run] = r
            
        if use_gm_mask:
            data['gm_mask'] = gm_mask
        
        n_comp_runs = len(data['funcs'])
        self.logger.info(f"Found {n_comp_runs} complete runs for task {task}.")

        ts = Taskset(task=task, space=space, n_jobs=self.n_jobs, log_level=self.logger.level, **data)
        setattr(self, task, ts)

    def visualize(self, statmap, space='fsnative', threshold=0, hemi='R', mask=None, **plot_args):
        """ Visualizes a statmap on an image/surface background. """
        if 'fs' in space and isinstance(statmap, nib.Nifti1Image):
            raise ValueError("Statmap is an image, but space is fs* something ...")
        
        bg = 'surface' if 'fs' in space else 'volume'
        self.logger.info(f"Visualizing statmap on {bg} ...")
        if 'fs' in space:
            fs_base = op.dirname(self.fs_dir)
            if space == 'fsnative':
                fs_id = f'sub-{self.sub}'
            else:
                fs_id = space
                    
            return plotting.view_surf(
                surf_mesh=op.join(fs_base, fs_id, 'surf', f'{hemi.lower()}h.inflated'),
                surf_map=statmap,
                bg_map=op.join(fs_base, fs_id, 'surf', f'{hemi.lower()}h.sulc'),
                threshold=threshold,
                **plot_args
            )
        else:
            
            if mask is not None:
                statmap = masking.unmask(statmap, mask)

            if space == 'T1w':
                bg = op.join(self.fp_dir, 'anat', f'sub-{self.sub}_desc-preproc_T1w.nii.gz')
            else:
                bg = 'MNI152'

                
            return plotting.view_img(
                stat_map_img=statmap,
                bg_img=bg,
                threshold=threshold,
                **plot_args
            )
        
    def plot_hrfs(self, taskset, rsq_cutoff=None, mask=None, condition=None):
        
        if not hasattr(getattr(self, taskset), 'rf'):
            raise ValueError(f"No HRF estimation has been done on taskset {taskset}!")
        
        if rsq_cutoff is None and mask is None:
            raise ValueError("Set either rsq_cutoff or mask!")
        
        if rsq_cutoff is not None and mask is not None:
            raise ValueError("Cannot set both rsq cutoff and mask!")
        
        fig, ax = plt.subplots(nrows=2, figsize=(15, 10), sharex=True)
        rf = getattr(self, taskset).rf
        rfv = rf.get_timecourses()
        
        if condition is not None:
            rfv = rfv.loc[rfv.index.get_level_values(0) == condition, :]
        
        if rsq_cutoff is not None:
            rsq = np.squeeze(rf.get_rsq())
            dat = rfv.values[:, rsq > rsq_cutoff]
        else:
            dat = rfv.values[:, mask]

        self.logger.info(f"Plotting {dat.shape[1]} voxels ...")
        t = rfv.index.get_level_values(2).values
        ax[0].plot(t, dat, lw=0.5)
        miny = ax[0].get_ylim()[0]
        ax[0].set_ylim(ax[0].get_ylim())
        for i in range(dat.shape[1]):
            tmax = t[dat[:, i].argmax()]
            ax[0].plot([tmax, tmax], [miny, dat[:, i].max()], ls='--', lw=0.1, c='k')

        ax[0].locator_params(axis='x', nbins=32)
        ax[0].set_xlim(0, 16)
        ax[1].set_xlabel("Time (seconds)", fontsize=20)
        ax[0].set_ylabel("Activity (A.U.)", fontsize=20)
        ax[1].set_ylabel("Frequency", fontsize=20)
        sns.distplot(t[dat.argmax(axis=0)], ax=ax[1])
        sns.despine()


class Taskset:
    
    def __init__(self, task, space, funcs, confs, events, ricors, mask, gm_mask=None, n_jobs=1, log_level=logging.INFO):
        """ Initializes a taskset object. 
        
        Parameters
        ----------
        task : str
            Name of task
        space : str
            Name of space (e.g., T1w, fsnative)
        funcs : list
            List of func files
        confs : list
            List of conf files
        events : list
            List of event files
        ricors : list
            List of retroicor files
        mask : Nifti1Image or None
            Mask for volume files, or None when dealing with surface files
        n_jobs : int
            Number of jobs to use for computations
        """
        self.task = task
        self.space = space
        self.funcs = funcs
        self.confs = confs
        self.events = events
        self.ricors = ricors
        self.mask = mask
        self.gm_mask = gm_mask
        self.n_jobs = n_jobs
        self.n_runs = len(self.funcs)
        self.preprocessed = False
        self.glm = None
        self.tr = []
        self.logger = logging.getLogger(task)
        self.logger.setLevel(log_level)

    def __repr__(self):
        msg = f"{self.task} taskset containg data of {self.n_runs} runs"
        return msg
    
    def preprocess(self, smoothing_fwhm=None, hp_cutoff=None, add_motion_outliers=True, add_ricor=True,
                   conf_vars=None, df_filter=None, slice_time_ref=0.5, regress_confounds=False):
        
        self.logger.info(f"Starting preprocessing for task {self.task}.")
        if conf_vars is None:
            conf_vars = []

        if 'fs' in self.space:
            for i in range(len(self.funcs)):
                this_tr = nib.load(self.funcs[i]).darrays[0].metadata['TimeStep']
                self.tr.append(np.float(this_tr) / 1000)
        else:
            self.tr = [nib.load(self.funcs[i]).header['pixdim'][4]
                       for i in range(len(self.funcs))]
            
        self.tr = np.array(self.tr)
        if not np.all(self.tr[0] == self.tr):
            self.logger.warning("Not all TRs across runsare the same ({self.tr})!")
        
        if add_ricor and len(self.ricors) == 0:
            self.logger.warning("No ricor file, so setting add_ricor to False.")
            add_ricor = False
        
        out = Parallel(n_jobs=self.n_jobs)(delayed(_run_preproc_in_parallel)(
                run=run,
                event=self.events[run],
                conf=self.confs[run],
                func=self.funcs[run],
                ricor=self.ricors[run] if add_ricor else None,
                task=self.task,
                space=self.space,
                conf_vars=conf_vars,
                mask=self.mask,
                smoothing_fwhm=smoothing_fwhm,
                hp_cutoff=hp_cutoff,
                add_motion_outliers=add_motion_outliers,
                df_filter=df_filter,
                tr=self.tr[run],
                slice_time_ref=slice_time_ref,
                logger=self.logger
            ) for run in range(self.n_runs)
        )
        
        prep_out = dict(
            func_ts=[o[0] for o in out],
            confs=[o[1] for o in out],
            events=[o[2] for o in out]
        )        
        
        if regress_confounds:
            clean = []
            for i, func in enumerate(prep_out['func_ts']):
                self.logger.info(f"Regressing out confounds for run {i+1} ...")
                sig = signal.clean(
                    signals=func,
                    confounds=prep_out['confs'][i].values,
                    standardize=False,
                    detrend=False
                )
                clean.append(sig)
            prep_out['func_clean'] = clean
        
        self.preprocessed = True
        self.preproc = prep_out    
    
    def compute_tsnr(self, run=None, mean_only=False, std_only=False):
        """ Computes tsnr (or mean/std) image of the (mean) functional data. """

        self.logger.info("Computing TSNR.")
        if mean_only and std_only:
            raise ValueError("Cannot return mean only and std only!")

        if run is None:
            n_runs = len(self.preproc['func_ts'])
        else:
            n_runs = 1
        
        stat = np.zeros((n_runs, self.preproc['func_ts'][0].shape[-1]))
        for run in range(n_runs):
            func = self.preproc['func_ts'][run]
            if mean_only:
                this_stat = func.mean(axis=0)
            elif std_only:
                this_stat = func.std(axis=0)
            else:
                this_stat = func.mean(axis=0) / func.std(axis=0)
            
            this_stat[np.isnan(this_stat)] = 0
            stat[run, :] = this_stat
        
        mean_stat = stat.mean(axis=0)
        if self.mask is not None:
            mean_stat = masking.unmask(mean_stat, self.mask)
        
        return mean_stat
    
    def glm_detect(self, dm=None, hrf_model='glover', noise_model='ols', osf=30,
                   slice_time_ref=0.5, mask=None, rf_condition=None):

        if not self.preprocessed:
            raise ValueError("Data was not preprocessed yet!")

        self.logger.info(f"Starting GLM estimation for task {self.task}.")

        glm_out = Parallel(n_jobs=self.n_jobs)(delayed(_run_glm_in_parallel)(
            dm=dm,
            run=run,
            event=self.preproc['events'][run],
            conf=self.preproc['confs'][run],
            func=self.preproc['func_ts'][run],
            hrf_model=hrf_model,
            noise_model=noise_model,
            tr=self.tr[run],
            osf=osf,
            slice_time_ref=slice_time_ref,
            mask=mask,
            rf_condition=rf_condition,
            logger=self.logger
            ) for run in range(self.n_runs)
        )

        self.glm = dict(
            labels=[run[0] for run in glm_out],
            results=[run[1] for run in glm_out],
            dms=[run[2] for run in glm_out],
            mask=mask
        )
        
        if len(glm_out[0]) == 4:
            self.glm['hrfs'] = [run[3] for run in glm_out]

    def shape_estimation(self, conf_vars=None, osf=30, slice_time_ref=0.5,
                         separate_conditions=False, **rf_args):
        
        if not self.preprocessed:
            raise ValueError("Data was not preprocessed yet!")

        self.logger.info(f"Starting HRF estimation for task {self.task}.")
        func_cleans, concat_events = [], []
        for run in range(self.n_runs):
            this_conf = self.preproc['confs'][run].values
            func_clean = signal.clean(
                signals=self.preproc['func_ts'][run],
                confounds=this_conf,
                detrend=False,
                standardize=True,
                t_r=self.tr[run]
            )
            func_cleans.append(func_clean)
            
            event = self.preproc['events'][run].copy()  
            # copy, otherwise shape_estimation interferes with glm_detect
            event.loc[:, 'onset'] += run * this_conf.shape[0] * TR
            concat_events.append(event)

        func_concat = np.concatenate(func_cleans)
        event_concat = pd.concat(concat_events, axis=0)#self.preproc['events'], axis=0)

        if not np.all(self.tr[0] == np.array(self.tr)):
            self.logger.warning("Not all TRs are the same, but running ResponseFitter on concat signal!")
        
        rf = ResponseFitter(
            input_signal=func_concat,
            sample_rate=1/self.tr[0],
            add_intercept=True,
            slice_time_ref=slice_time_ref,
            oversample_design_matrix=osf
        )
        
        if 'interval' not in rf_args.keys():
            rf_args['interval'] = [0, 20]
        
        if 'basis_set' not in rf_args.keys():
            rf_args['basis_set'] = 'fourier'
            
        if 'n_regressors' not in rf_args.keys():
            rf_args['n_regressors'] = 6
        
        if separate_conditions:
            for con in event_concat.trial_type.unique():
                onsets = event_concat.query('trial_type == @con').onset
                rf.add_event(con, onsets, **rf_args)
        else:
            rf.add_event('stim', event_concat.onset, **rf_args)

        rf.fit()
        self.rf = rf

    def compute_fxe_contrast(self, contrast_def, stat_type='t', run=None, output_type='z_score'):
        """ Computes a fixed effect across multiple runs. """
        
        self.logger.info(f"Computing contrast: {contrast_def} for task {self.task} ...")
        if self.glm is None:
            raise ValueError("GLM has not been run yet!")

        if run is None:
            results = self.glm['results']
            labels = self.glm['labels']
            dms = self.glm['dms']
            design_info = DesignInfo(dms[0].columns.tolist())
        else:
            results = self.glm['results'][run]
            labels = self.glm['labels'][run]
            dms = self.glm['dms'][run]
            design_info = DesignInfo(dms.columns.tolist())

        if isinstance(contrast_def, (np.ndarray, str)):
            con_vals = [contrast_def]
        elif isinstance(contrast_def, (list, tuple)):
            con_vals = contrast_def
        else:
            raise ValueError('contrast_def must be an array or str or list of'
                             ' (array or str)')

        for cidx, con in enumerate(con_vals):
            if not isinstance(con, np.ndarray):
                con_vals[cidx] = design_info.linear_constraint(con).coefs

        if run is None:
            contrast = _fixed_effect_contrast(labels, results, con_vals, stat_type)
        else:
            contrast = compute_contrast(labels, results, con_vals, stat_type)

        values = getattr(contrast, output_type)()
        if self.mask is not None:
            return masking.unmask(values, self.mask)
        else:
            return values
        
    def plot_design(self, exclude_confs=True):
        n_runs = len(self.glm['dms'])
        fig, ax = plt.subplots(nrows=n_runs, figsize=(15, 4 * n_runs))
        for run in range(n_runs):
            
            if n_runs != 1:
                this_ax = ax[run]
            else:
                this_ax = ax
            
            this_dm = self.glm['dms'][run].drop('constant', axis=1)
            if exclude_confs:
                this_dm = this_dm.drop(self.preproc['confs'][run].columns, axis=1)

            this_ax.plot(this_dm.values)
            this_ax.set_title(f"Run {run+1}")
            if run == 0:
                this_ax.legend(this_dm.columns)
        

def _run_preproc_in_parallel(run, event, conf, func, ricor, task, space, conf_vars, mask, hp_cutoff,
                             add_motion_outliers, smoothing_fwhm, df_filter, slice_time_ref, tr, logger):
    
    logger.info(f"Preprocessing run {run+1} ...")
    
    if 'fs' in space:
        func_ts = np.vstack([d.data[np.newaxis, :] for
                             d in nib.load(func).darrays])
    else:                       
        if not np.all(nib.load(func).affine == mask.affine):
            print(f"Resampling run {run+1} to mask affine, because they are different")
            func = image.resample_img(func, target_affine=mask.affine, target_shape=mask.shape)
                                                               
        func_ts = masking.apply_mask(
            imgs=func,
            mask_img=mask,
            smoothing_fwhm=smoothing_fwhm
        )

    n_vols = func_ts.shape[0]
    start_time = slice_time_ref * tr
    end_time = (n_vols - 1 + slice_time_ref) * tr
    frame_times = np.linspace(start_time, end_time, n_vols)

    event = pd.read_csv(event, sep='\t')
    if df_filter is not None: 
        event = df_filter(event)

    conf = pd.read_csv(conf, sep='\t')    
    all_conf_vars = conf_vars

    if add_motion_outliers:
        mo = [col for col in conf.columns if 'motion_outlier' in col]
        if len(mo) > 0:
            logger.info(f"Adding {len(mo)} motion outliers to design for run {run+1}.")
        all_conf_vars += mo

    if hp_cutoff is None:
        all_conf_vars += [col for col in conf.columns if 'cosine' in col]
        conf = conf.loc[:, all_conf_vars].fillna(0)
        conf = pd.concat((conf, cos_df), axis=1)
    else:
        conf = conf.loc[:, all_conf_vars].fillna(0)
        cos = _cosine_drift(hp_cutoff, frame_times)[:, :-1]
        cos_names = [f'cosine{ic}'.zfill(3) for ic in range(cos.shape[1])]
        cos_df = pd.DataFrame(data=cos, columns=cos_names, index=conf.index)
        conf = pd.concat((conf, cos_df), axis=1)
    
    if ricor is not None:
        ricor_df = pd.read_csv(ricor, sep='\t')
        conf = pd.concat((conf, ricor_df), axis=1)
    
    return func_ts, conf, event


def _run_glm_in_parallel(dm, run, event, conf, func, hrf_model, noise_model, tr, osf, slice_time_ref, 
                         mask, rf_condition, logger):

    logger.info(f"Fitting GLM to run {run+1} ...")
    n_vols = func.shape[0]
    start_time = slice_time_ref * tr
    end_time = (n_vols - 1 + slice_time_ref) * tr
    frame_times = np.linspace(start_time, end_time, n_vols)

    if mask is not None:
        func = func[:, mask.ravel()]

    if dm is not None:
        logger.info("Design-matrix was supplied, so fitting GLM immediately.")
        dm.index = frame_times
        conf.index = dm.index
        glm_results = run_glm(func, dm.values, noise_model=noise_model)
        return glm_results[0], glm_results[1], dm

    if not isinstance(hrf_model, str):  # custom HRF!
        logger.info("Using custom HRF-model.")
        conds = sorted(event.trial_type.unique())
        cols = ['constant'] + conds

        if isinstance(hrf_model, np.ndarray):
            # It's a SINGLE HRF (not a voxel-specific)
            X = np.zeros((n_vols, len(conds) + 1))
            X[:, 0] = 1  # intercept

            for i, con in enumerate(conds):  # loop over conditions
                trials = event.query('trial_type == @con')
                exp_cond = trials.loc[:, ['onset', 'duration', 'weight']]
                exp_cond['weight'] = 1

                # Upsample
                x, hr_frame_times = _sample_condition(exp_cond.values.T, frame_times, oversampling=osf)

                # Convolve predictor
                xconv = np.convolve(x, hrf_model)[:x.shape[0]]

                # Downsample
                f_interp = interp1d(hr_frame_times, xconv)
                X[:, i+1] = f_interp(frame_times)

            # Save the design matrix
            dm = pd.DataFrame(data=X, columns=cols, index=frame_times)

        elif isinstance(hrf_model, ResponseFitter):

            # Assuming it's a per-voxel GLM
            hrf_tc = hrf_model.get_timecourses()
            if rf_condition is not None:
                hrf_tc = hrf_tc.loc[hrf_tc.index.get_level_values(0) == rf_condition, :]

            tlength = hrf_tc.index.get_level_values(-1).values[-1]
            hrf_values = hrf_tc.values

            n_vox = func.shape[1]
            labels, results = [], {}
            canon = glover_hrf(tr=TR, oversampling=osf, time_length=tlength, onset=0.0)
            #canon /= canon.max()

            hrfs = np.zeros((3, canon.size, n_vox))
            for vox in range(n_vox):
                
                if vox % 1000 == 0:
                    print(f"Voxel {vox} / {n_vox} (run {run + 1})")
                
                this_hrf = hrf_values[:, vox]
                #this_hrf /= this_hrf.max()
                hrfs[0, :, vox] = this_hrf
                hrfs[1, :, vox] = canon
                #hrfs[2, :, vox] = this_hrf

                X = np.zeros((n_vols, len(conds) + 1))
                X[:, 0] = 1  # icept
                for i, con in enumerate(conds):
                    trials = event.query('trial_type == @con')
                    exp_cond = trials.loc[:, ['onset', 'duration', 'weight']]
                    exp_cond['weight'] = 1
                    x, hr_frame_times = _sample_condition(exp_cond.values.T, frame_times, oversampling=osf)

                    xconv = np.convolve(x, this_hrf)[:x.shape[0]]
                    #xconv = np.convolve(x, canon)[:x.shape[0]]
                    f_interp = interp1d(hr_frame_times, xconv)
                    X[:, i+1] = f_interp(frame_times)

                X = pd.DataFrame(data=X, columns=cols, index=frame_times)
                conf.index = X.index
                dm = pd.concat((X, conf), axis=1)
                lab, res = run_glm(func[:, vox, np.newaxis], dm.values, noise_model=noise_model)
                labels, results = _merge_regression_results(lab[0], res, labels, results, n_vox=n_vox)
            return np.array(labels), results, dm, hrfs
        else:
            raise ValueError("Unknown type for hrf_model; don't know what to do with it!")
    else:
        logger.info(f"Using default Nistats HRF model '{hrf_model}'.")
        dm = make_first_level_design_matrix(
            frame_times=frame_times,
            events=event,
            drift_model=None,
            hrf_model=hrf_model,
            fir_delays=None
        )

    conf.index = dm.index
    dm = pd.concat((dm, conf), axis=1)
    glm_results = run_glm(func, dm.values, noise_model=noise_model)
    return glm_results[0], glm_results[1], dm


def _merge_regression_results(lab, res, labels, results, n_vox):
    
    labels.append(lab)
    if lab not in results:
        results[lab] = res[lab]
        for attr in ['norm_resid', 'resid']:
            setattr(results[lab], attr, getattr(results[lab], attr).squeeze())
    else:
        for attr1d in ['SSE', 'logL', 'dispersion']:
            existing = getattr(results[lab], attr1d)
            new = getattr(res[lab], attr1d)
            concat = np.append(existing, new)
            setattr(results[lab], attr1d, concat)
            
        for attr2d in ['Y', 'norm_resid', 'resid', 'theta', 'wY', 'wresid']:    
            existing = getattr(results[lab], attr2d)
            new = getattr(res[lab], attr2d)
            concat = np.c_[existing, new]
            setattr(results[lab], attr2d, concat)
            
    return labels, results


def compute_rfx_contrast(imgs, design_matrix, contrast_def, mask=None, noise_model='ols', stat_type='t', output_type='z_score'):

    design_info = DesignInfo(design_matrix.columns.tolist())
    if isinstance(imgs, list):
        Y = np.stack([i.get_data() for i in imgs]).reshape(len(imgs), -1)        
    elif isinstance(imgs, np.ndarray):
        Y = imgs
    else:
        raise ValueError(f"Unknown format for Y ({type(imgs)}).")

    X = design_matrix.values
    labels, results = run_glm(Y, X, noise_model=noise_model)

    if isinstance(contrast_def, (np.ndarray, str)):
        con_vals = [contrast_def]
    elif isinstance(contrast_def, (list, tuple)):
        con_vals = contrast_def
    else:
        raise ValueError('contrast_def must be an array or str or list of'
                         ' (array or str)')

    for cidx, con in enumerate(con_vals):
        if not isinstance(con, np.ndarray):
            con_vals[cidx] = design_info.linear_constraint(con).coefs

    contrast = compute_contrast(labels, results, con_vals, stat_type)

    values = getattr(contrast, output_type)()
    if isinstance(imgs, list):
        values = nib.Nifti1Image(values.reshape(imgs[0].shape), affine=imgs[0].affine)

    return values


def plot_surface(data, mesh='infl', hemi='right', bg='sulc', space='fsaverage5', threshold=0):
    fs = datasets.fetch_surf_fsaverage(mesh=space, data_dir=None)
    return plotting.view_surf(
        surf_mesh=getattr(fs, f'{mesh}_{hemi}'),
        surf_map=data,
        bg_map=getattr(fs, f'{bg}_{hemi}'),
        threshold=threshold
    )
