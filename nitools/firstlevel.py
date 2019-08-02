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

    def create_taskset(self, task, space, acq=None, ses=False, hemi='R', reference_run=-1, use_gm_mask=False,
                       gm_threshold=0):
        """ Creates a 'taskset'. 
        
        parameters
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
        ts = Taskset(task=task, space=space, n_jobs=self.n_jobs, **data)
        setattr(self, task, ts)

    def visualize(self, statmap, space='fsnative', threshold=0, hemi='R', **plot_args):
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
            rsq = rf.get_rsq()
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
    
    def __init__(self, task, space, funcs, confs, events, ricors, mask, gm_mask=None, n_jobs=1):
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

    def __repr__(self):
        msg = f"{self.task} taskset containg data of {self.n_runs} runs"
        return msg
    
    def preprocess(self, smoothing_fwhm=None, conf_vars=None, df_filter=None, regress_confounds=False):
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
                df_filter=df_filter
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
            
        return stat.mean(axis=0)
    
    def glm_detect(self, hrf_model='glover', noise_model='ols', osf=30, TR=1.317, mask=None, rf_condition=None):

        if not self.preprocessed:
            raise ValueError("Data was not preprocessed yet!")

        print(f"Running a GLM for task {self.task} ...")
        glm_out = Parallel(n_jobs=self.n_jobs)(delayed(_run_glm_in_parallel)(
            run=run,
            event=self.preproc['events'][run],
            conf=self.preproc['confs'][run],
            func=self.preproc['func_ts'][run],
            hrf_model=hrf_model,
            noise_model=noise_model,
            TR=TR,
            osf=osf,
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

    def shape_estimation(self, smoothing_fwhm=None, conf_vars=None, osf=30, TR=1.317,
                         separate_conditions=False, **rf_args):
        
        if not self.preprocessed:
            raise ValueError("Data was not preprocessed yet!")

        print(f"Estimating HRF shape for task {self.task} ...")
        for run in range(self.n_runs):
            event, conf, func = self.preproc['events'][run], self.preproc['confs'][run], self.preproc['func_ts'][run]
            event.loc[:, 'onset'] += run * conf.shape[0] * TR
            self.preproc['events'][run] = event

        func_concat = np.concatenate(self.preproc['func_ts'])
        conf_concat = pd.concat(self.preproc['confs'], axis=0)
        event_concat = pd.concat(self.preproc['events'], axis=0)

        ses_id = np.concatenate(
            [[i+1] * self.preproc['func_ts'][i].shape[0]
            for i in range(self.n_runs)]
        )

        func_clean = signal.clean(
            signals=func_concat,
            sessions=ses_id,
            confounds=conf_concat.values,
            detrend=False,
            standardize=True,
            t_r=TR
        )
        
        rf = ResponseFitter(
            input_signal=func_clean,
            sample_rate=1/TR,
            add_intercept=True,
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

        rf.regress()
        self.rf = rf

    def compute_fxe(self, contrast_def, stat_type='t', run=None):
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

        z = contrast.z_score()
        if self.mask is not None:
            return masking.unmask(z, self.mask)
        else:
            return z
        

def _run_preproc_in_parallel(run, event, conf, func, task, space, conf_vars, mask, smoothing_fwhm, df_filter):
    print(f"\tPreparing run {run+1} ...")
    event = pd.read_csv(event, sep='\t')
    if df_filter is not None:
        event = df_filter(event)
    
    conf = pd.read_csv(conf, sep='\t')
    all_conf_vars = conf_vars + [col for col in conf.columns if 'cosine' in col]
    conf = conf.loc[:, all_conf_vars].fillna(0)

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

    return func_ts, conf, event


def _run_glm_in_parallel(run, event, conf, func, hrf_model, noise_model, TR, osf, mask, rf_condition):
    print(f"\tFitting run {run+1} ...")
    frame_times = np.arange(0, conf.shape[0] * TR, TR)
    
    if mask is not None:
        func = func[:, mask.ravel()]
    
    if not isinstance(hrf_model, str):  # custom HRF!
        n_vols = func.shape[0]
        conds = event.trial_type.unique()
        hr_frame_times = np.arange(0, TR * n_vols, TR / osf)[:-osf+1]
        cols = ['constant'] + conds.tolist()
        
        if isinstance(hrf_model, np.ndarray):
            X = np.zeros((n_vols, len(conds) + 1))
            X[:, 0] = 1
            for i, con in enumerate(conds):
                trials = event.query('trial_type == @con')
                x = np.zeros(hr_frame_times.size)
                for ii in range(trials.shape[0]):
                    row = trials.iloc[ii, :]
                    start = np.argmin(np.abs(hr_frame_times - row.onset))
                    end = start + int(row.duration / (TR / osf))
                    x[start:end] = 1
                xconv = np.convolve(x, hrf_model)[:x.shape[0]]
                f_interp = interp1d(hr_frame_times, xconv)
                X[:, i+1] = f_interp(frame_times)
            dm = pd.DataFrame(data=X, columns=cols, index=frame_times)
        elif isinstance(hrf_model, ResponseFitter):
            # Assuming it's a per-voxel GLM
            
            corrs = np.sqrt(hrf_model.get_rsq())
            corr_max = corrs.max()
            hrf_ts = hrf_model.get_timecourses()
            if rf_condition is not None:
                hrf_ts = hrf_ts.loc[hrf_ts.index.get_level_values(0) == rf_condition, :]

            tlength = hrf_ts.index.get_level_values(-1).values[-1]
            hrf_values = hrf_ts.values[:-1, :]
            
            n_vox = func.shape[1]
            labels, results = [], {}
            canon = glover_hrf(tr=TR, oversampling=osf, time_length=tlength, onset=0.0)
            canon /= canon.max()
            hrfs = np.zeros((3, canon.size, n_vox))
            for vox in tqdm_notebook(range(n_vox), desc=f'run {run+1}'):
                this_hrf = hrf_values[:, vox]
                this_hrf /= this_hrf.max()
                hrfs[0, :, vox] = this_hrf
                hrfs[1, :, vox] = canon

                this_corr = corrs[vox]
                if this_corr < 0:
                    this_corr = 0
                
                this_hrf = (this_hrf * (this_corr / corr_max) + canon * ((1 - corr_max) / corr_max)) / 2 
                hrfs[2, :, vox] = this_hrf
                hrfs[:, vox]
                X = np.zeros((n_vols, len(conds) + 1))
                X[:, 0] = 1  # icept
                for i, con in enumerate(conds):
                    trials = event.query('trial_type == @con')
                    x = np.zeros(hr_frame_times.size)
                    for ii in range(trials.shape[0]):
                        row = trials.iloc[ii, :]
                        start = np.argmin(np.abs(hr_frame_times - row.onset))
                        end = start + int(row.duration / (TR / osf))
                        x[start:end] = 1
                    xconv = np.convolve(x, this_hrf)[:x.shape[0]]
                    f_interp = interp1d(hr_frame_times, xconv)
                    X[:, i+1] = f_interp(frame_times)
                
                lab, res = run_glm(func[:, vox, np.newaxis], X, noise_model=noise_model)
                labels, results = _merge_regression_results(lab[0], res, labels, results, n_vox=n_vox)
            dm = pd.DataFrame(data=X, columns=cols, index=frame_times)
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

    cosine_drift[:, -1] = 1.
    return cosine_drift


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
