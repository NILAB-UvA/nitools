import os.path as op
import numpy as np
import nibabel as nib
import pandas as pd
import seaborn as sns
from nistats.first_level_model import FirstLevelModel, make_first_level_design_matrix, run_glm
from nistats.contrasts import _fixed_effect_contrast, compute_contrast
from nistats.hemodynamic_models import glover_hrf
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


class Dataset:
    
    def __init__(self, bids_dir, sub, postfix=None, n_jobs=4):
        """ Initializes Analysis object.
        
        parameters
        ----------
        bids_dir : str
            Path to BIDS directory
        sub : str
            Subject-identifier (e.g., '01')
        postfix : str
            Postfix for fmriprep dir
        n_jobs : int
            Number of jobs to use for computations
        """
        self.sub = sub
        self.bids_dir = op.join(op.abspath(bids_dir), f'sub-{sub}')
        self.postfix = postfix if postfix is not None else ''
        self.n_jobs = n_jobs

        self.deriv_dir = op.join(bids_dir, 'derivatives')
        self.fp_dir = op.join(self.deriv_dir, 'fmriprep', f'sub-{sub}{self.postfix}')
        self.fs_dir = op.join(self.deriv_dir, 'freesurfer', f'sub-{sub}')

    def create_taskset(self, task, space, tr, acq=None, ses=False, hemi='R', reference_run=-1, use_gm_mask=True,
                       gm_threshold=0.5):
        """ Creates a 'taskset'. 
        
        parameters
        ----------
        task : str
            Name of task
        space : str
            Name of space to analyze data in (e.g., 'T1w', 'fsnative')
        tr : float
            Time to repetition
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
            this_fp_dir = op.join(self.fp_dir, f'ses-*')
            this_bids_dir = op.join(self.bids_dir, f'ses-*')
        else:
            this_fp_dir = self.fp_dir
            this_bids_dir = self.bids_dir

        base_str += f'_task-{task}'

        if acq is not None:
            base_mri_str = f'{base_str}_acq-{acq}'
        else:
            base_mri_str = base_str
        
        conf_str = '_desc-confounds_regressors.tsv'
        all_confs = sorted(glob(op.join(this_fp_dir, 'func', f'{base_mri_str}*{conf_str}')))
        
        if 'fs' in space:
            func_str = f'_space-{space}_hemi-{hemi}.func.gii'
        else:
            func_str = f'_space-{space}_desc-preproc_bold.nii.gz'
        
        all_funcs = sorted(glob(op.join(this_fp_dir, 'func', f'{base_mri_str}*{func_str}')))
        
        if 'fs' in space:
            mask = None  # no mask for surface files!
            gm_mask = None
        else:
            mask_str = f'_space-{space}_desc-brain_mask.nii.gz'
            all_masks = sorted(glob(op.join(this_fp_dir, 'func', f'{base_mri_str}*{mask_str}')))
            if not all_masks:
                raise ValueError(f"Could not find any masks with match {base_mri_str}*{mask_str}")
            
            tmp = nib.load(all_masks[reference_run])
            for i, mask in enumerate(all_masks):  # check if we need to resapmle
                if not np.all(nib.load(mask).affine == tmp.affine):
                    print(f"WARNING: mask of run {i} has a different affine than "
                          "the reference run {reference_run}! Going to resample ...")
                    all_masks[i] = image.resample_img(
                        mask,
                        target_affine=tmp.affine,
                        target_shape=tmp.shape,
                        interpolation='nearest'
                    )
                
            mask = masking.intersect_masks(all_masks, threshold=0.5)
            if use_gm_mask:
                gm_mask = op.join(self.fp_dir, 'anat', f'sub-{self.sub}_label-GM_probseg.nii.gz')
                gm_img = nib.load(gm_mask)
                gm_mask = nib.Nifti1Image((gm_img.get_data() > gm_threshold).astype(int), affine=gm_img.affine)
                
                gm_mask = image.resample_img(
                    gm_mask,
                    target_affine=tmp.affine,
                    target_shape=tmp.shape,
                    interpolation='nearest'
                )
                mask = masking.intersect_masks([mask, gm_mask], threshold=1)

        event_str = '_events.tsv'
        all_events = sorted(glob(op.join(this_bids_dir, 'func', f'{base_mri_str}*{event_str}')))
        if not all_events:
            raise ValueError(f"Could not find any events with match {base_str}*{event_str}")
        
        if len(all_funcs) != len(all_events) != len(all_confs):
            warn(f"Found {len(all_funcs)} funcs but {len(all_events)} events and {len(all_confs)} confs!")

        ricor_str = '_ricor.txt'
        data = dict(funcs=dict(), events=dict(), confs=dict(), ricors=dict(), mask=mask)
        for run, (func, event, conf) in enumerate(zip(all_funcs, all_events, all_confs)):
            data['funcs'][run] = func
            data['events'][run] = event
            data['confs'][run] = conf

            ricor_f = func.replace(func_str, ricor_str)
            if op.isfile(ricor_f):
                data['ricor'][run] = ricor_f
        
        if use_gm_mask:
            data['gm_mask'] = gm_mask
        
        n_comp_runs = len(data['funcs'])
        print(f"Found {n_comp_runs} complete runs for task {task}")
        ts = Taskset(tr=tr, task=task, space=space, n_jobs=self.n_jobs, **data)
        setattr(self, task, ts)

    def visualize(self, statmap, space='fsnative', threshold=0, hemi='R', mask=None, **plot_args):
        """ Visualizes a statmap on an image/surface background. """
        if 'fs' in space and isinstance(statmap, nib.Nifti1Image):
            raise ValueError("Statmap is an image, but space is fs* something ...")
        
        bg = 'surface' if 'fs' in space else 'volume'
        print(f"Visualizing statmap on {bg} ...")
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
                bg = None

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

        print(f"Plotting {dat.shape[1]} voxels ...")
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
    
    def __init__(self, tr, task, space, funcs, confs, events, ricors, mask, gm_mask=None, n_jobs=1):
        """ Initializes a taskset object. 
        
        Parameters
        ----------
        tr : float
            Time to repetition
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
        self.tr = tr
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

    def __repr__(self):
        msg = f"{self.task} taskset containg data of {self.n_runs} runs"
        return msg
    
    def preprocess(self, smoothing_fwhm=None, hp_cutoff=None, conf_vars=None, df_filter=None,
                   slice_time_ref=0.5, regress_confounds=False):
        print(f"Preprocessing data for task {self.task} ...")
        if conf_vars is None:
            conf_vars = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
                
        out = Parallel(n_jobs=self.n_jobs)(delayed(_run_preproc_in_parallel)(
                run=run,
                event=self.events[run],
                conf=self.confs[run],
                func=self.funcs[run],
                task=self.task,
                space=self.space,
                conf_vars=conf_vars,
                mask=self.mask,
                smoothing_fwhm=smoothing_fwhm,
                hp_cutoff=hp_cutoff,
                df_filter=df_filter,
                tr=self.tr,
                slice_time_ref=slice_time_ref
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
                print(f"Regressing out confounds for run {i+1} ...")
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

        print(f"Running a GLM for task {self.task} ...")
        glm_out = Parallel(n_jobs=self.n_jobs)(delayed(_run_glm_in_parallel)(
            dm=dm,
            run=run,
            event=self.preproc['events'][run],
            conf=self.preproc['confs'][run],
            func=self.preproc['func_ts'][run],
            hrf_model=hrf_model,
            noise_model=noise_model,
            tr=self.tr,
            osf=osf,
            slice_time_ref=slice_time_ref,
            mask=mask,
            rf_condition=rf_condition
            ) for run in range(self.n_runs)
        )

        self.glm = dict(
            labels=[run[0] for run in glm_out],
            results=[run[1] for run in glm_out],
            dms=[run[2] for run in glm_out],
            funcs=self.preproc['func_ts'],
            confs=self.preproc['confs'],
            events=self.preproc['events'],
            mask=mask
        )
        
        if len(glm_out[0]) == 4:
            self.glm['hrfs'] = [run[3] for run in glm_out]

    def shape_estimation(self, conf_vars=None, osf=30, TR=1.317, slice_time_ref=0.5,
                         separate_conditions=False, **rf_args):
        
        if not self.preprocessed:
            raise ValueError("Data was not preprocessed yet!")

        print(f"Estimating HRF shape for task {self.task} ...")
        func_cleans, concat_events = [], []
        for run in range(self.n_runs):
            this_conf = self.preproc['confs'][run].values
            func_clean = signal.clean(
                signals=self.preproc['func_ts'][run],
                confounds=this_conf,
                detrend=False,
                standardize=True,
                t_r=self.tr
            )
            func_cleans.append(func_clean)
            
            event = self.preproc['events'][run].copy()  
            # copy, otherwise shape_estimation interferes with glm_detect
            event.loc[:, 'onset'] += run * this_conf.shape[0] * TR
            concat_events.append(event)

        func_concat = np.concatenate(func_cleans)
        event_concat = pd.concat(concat_events, axis=0)#self.preproc['events'], axis=0)

        rf = ResponseFitter(
            input_signal=func_concat,
            sample_rate=1/TR,
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

    def compute_fxe(self, contrast_def, stat_type='t', run=None, output_type='z_score'):
        """ Computes a fixed effect across multiple runs. """
        print(f"Computing contrast: {contrast_def} for task {self.task} ...")
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
                this_dm = this_dm.drop(self.glm['confs'][run].columns, axis=1)

            this_ax.plot(this_dm.values)
            this_ax.set_title(f"Run {run+1}")
            if run == 0:
                this_ax.legend(this_dm.columns)
        

def _run_preproc_in_parallel(run, event, conf, func, task, space, conf_vars, mask, hp_cutoff,
                             smoothing_fwhm, df_filter, slice_time_ref, tr):
    print(f"\tPreparing run {run+1} ...")
    
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

    mo = [col for col in conf.columns if 'motion_outlier' in col]
    if len(mo) > 0:
        print(f"\tAdding {len(mo)} motion outliers to design for run {run+1}...")
        all_conf_vars += mo

    if hp_cutoff is None:
        all_conf_vars += [col for col in conf.columns if 'cosine' in col]
        conf = conf.loc[:, all_conf_vars].fillna(0)
    else:
        conf = conf.loc[:, all_conf_vars].fillna(0)
        cos = _create_cosine_set(frame_times, period_cut=hp_cutoff)
        cos_names = [f'cosine{ic}'.zfill(3) for ic in range(cos.shape[1])]
        cos_df = pd.DataFrame(data=cos, columns=cos_names, index=conf.index)
        conf = pd.concat((conf, cos_df), axis=1)
            
    return func_ts, conf, event


def _run_glm_in_parallel(dm, run, event, conf, func, hrf_model, noise_model, tr, osf, slice_time_ref, 
                         mask, rf_condition):

    print(f"\tFitting GLM to run {run+1} ...")
    n_vols = func.shape[0]
    start_time = slice_time_ref * tr
    end_time = (n_vols - 1 + slice_time_ref) * tr
    frame_times = np.linspace(start_time, end_time, n_vols)

    if mask is not None:
        func = func[:, mask.ravel()]

    if dm is not None:
        dm.index = frame_times
        # skip to below
    elif not isinstance(hrf_model, str):  # custom HRF!
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
        dm = make_first_level_design_matrix(
            frame_times=frame_times,
            events=event,
            drift_model=None,
            hrf_model=hrf_model,
            fir_delays=np.arange(1, 10)
        )

    conf.index = dm.index
    dm = pd.concat((dm, conf), axis=1)
    glm_results = run_glm(func, dm.values, noise_model=noise_model)
    return glm_results[0], glm_results[1], dm


def _create_cosine_set(frame_times, period_cut=0.01):
    n_frames = len(frame_times)
    n_times = np.arange(n_frames)
    hfcut = 1. / period_cut  # input parameter is the period
    dt = frame_times[1] - frame_times[0]
    order = int(np.floor(2 * n_frames * hfcut * dt))
    # s.t. hfcut = 1 / (2 * dt) yields n_frames
    cosine_drift = np.zeros((n_frames, order + 1))
    normalizer = np.sqrt(2.0 / n_frames)

    for k in range(1, order + 1):
        cosine_drift[:, k - 1] = normalizer * np.cos(
            (np.pi / n_frames) * (n_times + .5) * k)

    #cosine_drift[:, -1] = 1.
    return cosine_drift[:, :-1]


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
