#!/usr/bin/env python3
"""Script 26: 12-method Localize-MI comparison, using Script 17 plotting style.

Two independent reporting cohorts:
 A complete_any_status: one observation for every one of 61 runs, FAIL allowed.
 B complete_all_pass: the same 61 observations must all be PASS.

For the complete_all_pass cohort, a method's primary selection parameter value
is eligible only if no run using that value failed anywhere in the unfiltered
loaded grid, and every observed configuration using it covers all reference
runs, even when other parameters differ. A value with a failure is excluded
even if it is the only observed value. Configurations containing a failed
primary value are excluded; each remaining full configuration must also pass
all 61 runs. Selection uses lowest configuration median, then more successful
runs, lower primary-parameter overall median, lower configuration mean, and ID.
The complete_any_status cohort remains a descriptive comparison that allows
FAIL rows; its medians must be interpreted with PASS coverage. Overall-grid
values are within-run medians across eligible PASS solutions, as in Script 17.

This is an in-sample descriptive comparison, not unseen-subject validation.
Script 25's 06/07 CSVs supply complete comparison rows. Script 24's unfiltered
master supplies reliability evidence before incomplete configurations are removed.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import re
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ['MNE','dSPM','sLORETA','eLORETA']
OTHER = ['Continuous-ECD','MxNE','RAP-MUSIC','LCMV']
NEW = ['irMxNE','Champagne','Gamma-MAP','SESAME']
METHODS = ORIGINAL + OTHER + NEW
NEW_NAMES = {'irmxne':'irMxNE','champagne':'Champagne','gamma_map':'Gamma-MAP','sesame':'SESAME'}
COLORS = {
    'MNE':'#4C78A8', 'dSPM':'#E45756', 'sLORETA':'#54A24B', 'eLORETA':'#8E5C99',
    'MxNE':'#72B7B2', 'irMxNE':'#2D6A9F',
    'Continuous-ECD':'#F58518', 'RAP-MUSIC':'#B279A2', 'LCMV':'#ECA82C',
    'Gamma-MAP':'#8561A8', 'Champagne':'#D65A31', 'SESAME':'#299C87',
}
METHOD_GROUPS = [
    ('Distributed / minimum-norm',['MNE','dSPM','sLORETA','eLORETA']),
    ('Sparse',['MxNE','irMxNE']),
    ('Dipole-based',['Continuous-ECD','RAP-MUSIC']),
    ('Beamformer',['LCMV']),
    ('Bayesian / hierarchical',['Gamma-MAP','Champagne','SESAME']),
]

# Reference publication years, not necessarily first invention dates.
# Continuous-ECD: 1985 is an established dipole-model paper, not an origin claim.
METHOD_YEARS = {
    'MNE': 1994, 'dSPM': 2000, 'sLORETA': 2002, 'eLORETA': 2007,
    'Continuous-ECD': 1985, 'MxNE': 2012, 'RAP-MUSIC': 1999,
    'LCMV': 1997, 'irMxNE': 2016, 'Champagne': 2010,
    'Gamma-MAP': 2009, 'SESAME': 2014,
}

def add_year_labels_to_figure(fig):
    """Edit display-only method labels; preserve all underlying data names."""
    labels = {m: f'{m} ({METHOD_YEARS[m]})' for m in METHODS}
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            # Force label Text objects to reflect their formatters before editing.
            for tick in axis.get_major_ticks():
                for item in (tick.label1, tick.label2):
                    value = item.get_text()
                    if value in labels:
                        item.set_text(labels[value])
            # FixedLocator + FixedFormatter avoids a redraw overwriting the text.
            ticks = axis.get_ticklocs()
            texts = [t.get_text() for t in axis.get_ticklabels()]
            if any(t in labels for t in texts) and len(ticks) == len(texts):
                axis.set_ticks(ticks, labels=[labels.get(t, t) for t in texts])
        legend = ax.get_legend()
        if legend is not None:
            for item in legend.get_texts():
                item.set_text(labels.get(item.get_text(), item.get_text()))
    for legend in fig.legends:
        for item in legend.get_texts():
            item.set_text(labels.get(item.get_text(), item.get_text()))



def arguments():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--script17', type=Path, default=ROOT/'scripts/17_compare_all_methods_selected_and_overall.py')
    a.add_argument('--original-grid', type=Path, default=ROOT/'outputs/grid_all_runs/grid_results.csv')
    a.add_argument('--script15-root', type=Path, default=ROOT/'outputs/15_full_additional_method_grid')
    a.add_argument('--mxne-extension-root', type=Path, default=ROOT/'outputs/15_mxne_alpha_extension',
                   help='Folder containing the newer MxNE alpha-extension grid_results CSVs.')
    a.add_argument('--mxne-extra-results', type=Path, action='append', default=[],
                   help='Explicit MxNE extension CSV(s); these replace the old MxNE grid. If omitted, discover extension CSVs under --mxne-extension-root and --script15-root.')
    a.add_argument('--new-any', type=Path, default=ROOT/'outputs/25_next_method_audit/06_complete_any_status.csv')
    a.add_argument('--new-pass', type=Path, default=ROOT/'outputs/25_next_method_audit/07_complete_all_pass.csv')
    a.add_argument('--new-master', type=Path, default=ROOT/'outputs/23_full_next_method_grid/23_all_next_methods_combined.csv',
                   help='Unfiltered Script-24 master, required for failure and coverage checks before Script-25 filtering.')
    a.add_argument('--output-root', type=Path, default=ROOT/'outputs/26_twelve_methods_comparison')
    a.add_argument('--expected-runs', type=int, default=61)
    a.add_argument('--dpi', type=int, default=240)
    a.add_argument('--no-figures', action='store_true')
    a.add_argument('--overwrite', action='store_true')
    return a.parse_args()


def script17_module(path):
    if not path.is_file():
        raise FileNotFoundError(f'Script 17 not found: {path}. Supply --script17 FILE')
    spec = importlib.util.spec_from_file_location('localizemi_script17', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.METHOD_ORDER = METHODS[:]
    mod.METHOD_COLORS.update(COLORS)
    return mod


def reference_keys(frame):
    return set(map(tuple, frame[['subject','run']].astype(str).drop_duplicates().to_numpy()))


def normalize(frame, label, mod):
    mod.require_columns(frame, ['method','subject','run','status','montage'], label)
    out = mod.standardize_grid_error_column(frame, label)
    out = mod.normalize_status(out)
    out['subject'] = out.subject.astype(str)
    out['run'] = out.run.astype(str)
    out['method'] = out.method.astype(str)
    if out.loc[out.status.eq('PASS'), 'localization_distance_mm'].isna().any():
        raise ValueError(f'{label}: PASS row with missing error')
    return out


def load_non_mxne_additional(args, mod):
    """Load the three Script-15 methods independently of Script-17's MxNE mode."""
    if hasattr(mod, 'additional_grid_paths'):
        available=mod.additional_grid_paths(args.script15_root)
    else:
        available={
            'Continuous-ECD':args.script15_root/'continuous_ecd'/'grid_results.csv',
            'RAP-MUSIC':args.script15_root/'rap_music'/'grid_results.csv',
            'LCMV':args.script15_root/'lcmv'/'grid_results.csv',
        }
    frames=[]
    for method in ['Continuous-ECD','RAP-MUSIC','LCMV']:
        path=Path(available[method])
        if not path.is_file():
            raise FileNotFoundError(f'Missing Script-15 {method} grid: {path}')
        frame=normalize(pd.read_csv(path,low_memory=False),str(path),mod)
        observed=set(frame.method.dropna().astype(str).str.strip())
        if observed!={method}:
            raise ValueError(f'Expected only {method} rows in {path}; found {observed}')
        frames.append(frame)
    return pd.concat(frames,ignore_index=True,sort=False)


def old_grid(args, mod):
    original = mod.load_original_grid(args.original_grid)
    # Read these explicitly: current Script 17 variants can make
    # load_additional_grids(..., mode='old') return only the MxNE grid.
    # MxNE is handled separately below so its old grid cannot leak in.
    additional=load_non_mxne_additional(args,mod)
    extra_paths=list(args.mxne_extra_results)
    baseline=(args.script15_root/'mxne'/'grid_results.csv').resolve()
    if not extra_paths:
        baseline_dir=(args.script15_root/'mxne').resolve()
        extension_root=args.mxne_extension_root.resolve()
        script15_root=args.script15_root.resolve()
        discovered=[]
        if extension_root.exists():
            discovered.extend(p for p in extension_root.rglob('grid_results*.csv')
                              if p.resolve()!=baseline)
        if script15_root.exists():
            for path in script15_root.rglob('grid_results*.csv'):
                resolved=path.resolve()
                # Script 15 also contains Continuous-ECD, RAP-MUSIC, and LCMV
                # grids. Only MxNE-named folders may supply MxNE extensions.
                if (resolved!=baseline and path.parent.resolve()!=baseline_dir
                        and 'mxne' in str(path.parent).lower()):
                    discovered.append(path)
        extra_paths=sorted(set(discovered))
    if not extra_paths:
        raise FileNotFoundError(
            'No MxNE extra-result CSVs found. Pass one or more --mxne-extra-results paths; '
            'the old mxne/grid_results.csv is intentionally excluded.')
    extras=[]
    for path in extra_paths:
        frame=pd.read_csv(path,low_memory=False)
        mod.require_columns(frame,['method','subject','run','status','montage'],str(path))
        observed=set(frame.method.dropna().astype(str).str.strip().str.lower())
        if observed!={'mxne'}:
            raise ValueError(f'Expected only MxNE rows in extra result {path}; found {observed}')
        frame=normalize(frame,str(path),mod)
        frame['method']='MxNE'
        if 'configuration_id' not in frame:
            param_cols=[c for c in frame.columns if re.search(
                r'(^|_)(alpha|loose|depth|snr|lambda2?)(_|$)',str(c),re.I)]
            if not param_cols:
                raise ValueError(f'{path}: MxNE extras need configuration_id or identifiable parameter columns')
            frame['configuration_id']=frame[param_cols].astype(str).agg('|'.join,axis=1)
        extras.append(frame)
    additional=pd.concat([additional,*extras],ignore_index=True,sort=False)
    return pd.concat([original,additional], ignore_index=True, sort=False)


def prepare_old(old, reference):
    """Identify full configurations for old methods, not row-specific outcomes."""
    controlled={
        'Continuous-ECD':['ecd_min_dist_mm'],
        'MxNE':['mxne_alpha','mxne_loose','mxne_depth','mxne_iterations',
                'alpha','loose','depth'],
        'RAP-MUSIC':['rap_music_n_dipoles'],
        'LCMV':['lcmv_reg','lcmv_window_half_ms','lcmv_data_covariance_method',
                'lcmv_pick_ori','lcmv_weight_norm','lcmv_reduce_rank'],
    }
    out=[]
    for method, group in old.groupby('method', sort=False):
        group=group.copy()
        if method in ORIGINAL:
            fields = [c for c in ['method_origin','montage','loose','depth','snr','lambda2'] if c in group.columns]
            screen_fields=fields[:]
        else:
            identity_fields=['montage','configuration_id']
            if not all(c in group.columns for c in identity_fields):
                raise ValueError(f'Cannot identify full configurations for {method}: {identity_fields}')
            # Script 15 writes the realized noise covariance estimator in
            # covariance_method. It can change between runs and MUST NOT be
            # part of a requested configuration. Use controlled inputs only.
            screen_fields=['montage']+[c for c in controlled.get(method,[])
                                       if c in group and group[c].notna().any()]
            # Failed fits can lack method inputs. Recover an input only when
            # its stable Script-15 ID + montage identifies one observed value.
            for col in screen_fields[1:]:
                group[col]=group[col].replace(r'^\s*$',np.nan,regex=True)
                known=group.groupby(identity_fields,dropna=False)[col]
                unique=known.transform('nunique')
                first=known.transform('first')
                fill=group[col].isna() & unique.eq(1)
                group.loc[fill,col]=first.loc[fill]
            # Include actual varied inputs in the identity; extension files may
            # reuse configuration_id values across different alpha settings.
            fields=list(dict.fromkeys(identity_fields+screen_fields))
        if not all(c in group for c in fields):
            raise ValueError(f'Cannot identify old configuration inputs for {method}: {fields}')
        part = group.copy()
        # Source grids use stable configuration identifiers; normalizing numeric
        # strings avoids splitting alpha=0.1 vs alpha=0.10.
        def norm(v):
            if pd.isna(v): return '<NA>'
            try:
                if not isinstance(v,str) or v.replace('.','',1).replace('-','',1).isdigit():
                    return f'{float(v):.12g}'
            except (TypeError,ValueError): pass
            return str(v).strip()
        cfg = part[fields].apply(lambda s:s.map(norm))
        part['configuration_key'] = (method + ':' + cfg.agg('|'.join,axis=1))
        # configuration_id is retained for provenance but is not itself a
        # tunable parameter in the global parameter-value reliability check.
        screen = part[screen_fields].apply(lambda s:s.map(norm))
        display=part[fields].apply(lambda s:s.map(norm))
        part['configuration_parameters'] = display.apply(
            lambda r:json.dumps(r.to_dict(), sort_keys=True),axis=1)
        part['screen_parameters'] = screen.apply(lambda r:json.dumps(r.to_dict(), sort_keys=True),axis=1)
        out.append(part)
    return pd.concat(out, ignore_index=True, sort=False)


def eligibility(grid, ref, require_all_pass):
    frames=[]; records=[]
    for (method, cfg), group in grid.groupby(['method','configuration_key'], sort=False, dropna=False):
        keys=reference_keys(group)
        counts=group.groupby(['subject','run'], dropna=False).size()
        # Duplicates must be resolved, never bias medians as extra solutions.
        if counts.max() > 1:
            same=group.groupby(['subject','run'],dropna=False).apply(
                lambda g:g[['status','localization_distance_mm']].fillna(-1).drop_duplicates().shape[0]==1,
                include_groups=False)
            if not same.all():
                raise ValueError(f'Conflicting duplicate observations: {method}/{cfg}')
            group=group.drop_duplicates(['subject','run']).copy()
        complete=(keys == ref and len(group)==len(ref))
        passed=int(group.status.eq('PASS').sum())
        eligible=complete and (not require_all_pass or passed==len(ref))
        records.append({'method':method,'configuration_key':cfg,
            'configuration_parameters':group.configuration_parameters.iloc[0],
            'runs':len(keys),'pass':passed,'fail':int(group.status.eq('FAIL').sum()),
            'eligible':eligible})
        if eligible: frames.append(group)
    if not frames: raise ValueError('No complete eligible configurations')
    return pd.concat(frames,ignore_index=True,sort=False),pd.DataFrame(records)


def load_new(path, expected_methods, ref, mod):
    data=normalize(pd.read_csv(path,low_memory=False),str(path),mod)
    data['method']=data.method.str.lower().str.replace('-','_',regex=False).map(NEW_NAMES)
    if set(data.method.dropna()) != set(expected_methods):
        raise ValueError(f'{path}: methods mismatch: {data.method.unique()}')
    for name in ['configuration_key','configuration_parameters']:
        if name not in data.columns: raise ValueError(f'{path}: missing {name}; rerun Script 25')
    data['screen_parameters'] = data.configuration_parameters
    grouped=data.groupby(['method','configuration_key'],dropna=False)
    for (method,cfg), group in grouped:
        if reference_keys(group)!=ref or len(group)!=len(ref):
            raise ValueError(f'{path}: configuration not exactly 61 reference runs: {method}/{cfg}')
        if group.duplicated(['subject','run']).any():
            raise ValueError(f'{path}: duplicate run in {method}/{cfg}')
    return data


PRIMARY_PARAMETER_HINTS = {
    'MxNE':['alpha'], 'irMxNE':['irmxne_alpha'],
    'Gamma-MAP':['gamma_alpha'], 'SESAME':['sesame_fs'],
    'Champagne':['champagne_pruning_thresh','champagne_prune'],
    'Continuous-ECD':['ecd_min_dist_mm','min_dist'],
}


def primary_parameter_for(method, group):
    params=[]
    for value in group.screen_parameters.drop_duplicates():
        try: params.append(json.loads(value))
        except (TypeError,json.JSONDecodeError): pass
    fields={key for item in params if isinstance(item,dict) for key in item}
    for hint in PRIMARY_PARAMETER_HINTS.get(method,[]):
        if hint in fields: return hint
    alpha_fields=sorted(f for f in fields if f.lower().endswith('alpha'))
    return alpha_fields[0] if alpha_fields else None


def load_new_reliability_evidence(path, mod):
    """Read raw allowed experiments; recover only primary values encoded by IDs.

    Never assign a failed observation to a different alpha from its stored ID.
    Other method inputs and all input CSVs remain unchanged.
    """
    allowed={
        'champagne':{'23_full_next_method_grid','23_champagne_pruning_extension'},
        'irmxne':{'23_full_next_method_grid','23_irmxne_alpha15_extension','23_irmxne_alpha20_extension'},
        'gamma_map':{'23_full_next_method_grid','23_gamma_alpha_extension','23_gamma_alpha_low_extension','23_gamma_xyz_true'},
        'sesame':{'23_full_next_method_grid','23_sesame_fs_extension'},
    }
    data=normalize(pd.read_csv(path,low_memory=False),str(path),mod)
    mod.require_columns(data,['source_experiment','configuration_id'],str(path))
    ids=data.method.str.lower().str.replace('-','_',regex=False)
    mask=pd.Series([s in allowed.get(m,set()) for m,s in zip(ids,data.source_experiment)],index=data.index)
    data=data.loc[mask].copy(); data['method']=ids.loc[mask].map(NEW_NAMES)
    params=[]
    for row in data.to_dict('records'):
        field=PRIMARY_PARAMETER_HINTS[row['method']][0]
        value=row.get(field,np.nan)
        if pd.isna(value):
            patterns={'Gamma-MAP':r'^gamma_a([0-9.eE+-]+)_',
                      'irMxNE':r'^irmxne_a([0-9.eE+-]+)_',
                      'SESAME':r'^sesame_fs_([0-9p.eE+-]+)',
                      'Champagne':r'^champagne_thresh_([0-9.eE+-]+)'}
            match=re.search(patterns[row['method']],str(row['configuration_id']))
            if not match:
                raise ValueError(f'Cannot establish primary parameter for raw row: {row["method"]}/{row["configuration_id"]}')
            value=float(match.group(1).replace('p','.'))
        params.append(json.dumps({field:float(value)}))
    data['screen_parameters']=params
    # IDs encode controlled grid settings. Normalize numeric spellings so
    # gamma_a0.50 and gamma_a0.5 describe the same requested configuration.
    def canonical_id(value):
        return re.sub(r'\d+(?:\.\d+)?',lambda m:format(float(m.group()),'.12g'),str(value))
    data['configuration_key']=(data.method+':'+data.montage.astype(str)+':'
                               +data.configuration_id.map(canonical_id))
    return data


def parameter_value_screen(grid, evidence=None, ref=None):
    """Exclude values with failures or incomplete observed grids in the evidence.

    The primary field is method-specific (e.g. alpha for MxNE and fs for
    SESAME). Other parameter dimensions are screened at the full-configuration
    level, where the selected configuration must pass all 61 runs.
    """
    rows=[]
    config_records=[]
    def canonical(v):
        if pd.isna(v): return '<NA>'
        if isinstance(v,(bool,np.bool_)): return str(bool(v)).lower()
        if isinstance(v,(int,float,np.integer,np.floating)):
            return f'{float(v):.12g}'
        return str(v).strip()
    parsed=[]
    for _, row in grid.iterrows():
        try:
            params=json.loads(row.screen_parameters)
        except (TypeError, json.JSONDecodeError):
            params={}
        parsed.append(params if isinstance(params,dict) else {})
    work=grid.copy()
    work['_screen_params']=parsed
    blocked=set()
    evidence_work=work if evidence is None else evidence.copy()
    if evidence is not None:
        evidence_work['_screen_params']=evidence_work.screen_parameters.map(json.loads)
    for method, group in evidence_work.groupby('method',sort=False):
        primary=primary_parameter_for(method,group)
        fields=[primary] if primary else []
        for field in fields:
            vals=group._screen_params.map(lambda p: p.get(field, '<MISSING>'))
            normalized=vals.map(canonical)
            distinct=normalized.nunique(dropna=False)
            fail_mask=group.status.eq('FAIL')
            failed=set(normalized.loc[fail_mask])
            incomplete={}
            if ref is not None:
                for cfg,g in group.assign(_value=normalized).groupby('configuration_key',sort=False):
                    if reference_keys(g)!=ref:
                        for value in g._value.unique():
                            incomplete[value]=incomplete.get(value,0)+1
            other=set(normalized.loc[~group.status.isin(['PASS','FAIL'])])
            counts=group.assign(_value=normalized).groupby('_value',dropna=False).agg(
                configurations=('configuration_key','nunique'),
                pass_rows=('status',lambda s:int(s.eq('PASS').sum())),
                fail_rows=('status',lambda s:int(s.eq('FAIL').sum()))).reset_index()
            for _, item in counts.iterrows():
                value=item['_value']
                varied=distinct>1
                bad=value in failed or incomplete.get(value,0)>0 or value in other
                rows.append({'method':method,'parameter':field,'value':value,
                    'varied_parameter':varied,'configurations':int(item.configurations),
                    'pass_rows':int(item.pass_rows),'fail_rows':int(item.fail_rows),
                    'incomplete_configurations':incomplete.get(value,0),
                    'screen_scope':'unfiltered loaded grid' if evidence is not None else 'complete configurations',
                    'exclusion_reason':'; '.join(reason for condition,reason in [
                        (value in failed,'recorded failure'),
                        (incomplete.get(value,0)>0,'incomplete configuration coverage'),
                        (value in other,'unrecognised status')] if condition),
                    'eligible_value':not bad})
                if bad: blocked.add((method,field,value))
    keep=[]
    for idx,row in work.iterrows():
        rejected=[]
        for field,value in row['_screen_params'].items():
            key=(row.method,field,canonical(value))
            if key in blocked: rejected.append(f'{field}={canonical(value)}')
        keep.append(not rejected)
        config_records.append({'method':row.method,'configuration_key':row.configuration_key,
            'configuration_parameters':row.configuration_parameters,
            'parameter_value_failures':'; '.join(sorted(set(rejected))),
            'eligible_after_parameter_value_screen':not rejected})
    filtered=work.loc[keep].drop(columns=['_screen_params']).copy()
    config_audit=pd.DataFrame(config_records).drop_duplicates(
        ['method','configuration_key'],keep='first')
    return filtered,pd.DataFrame(rows),config_audit


def select_and_overall(grid, methods, ref, mode):
    defs=[]; selected=[]
    for method in methods:
        m=grid.loc[grid.method.eq(method)]
        if m.empty: raise ValueError(f'{mode}: no eligible configurations: {method}')
        primary_field=primary_parameter_for(method,m)
        value_overall={}
        if primary_field:
            def field_value(raw):
                try: parsed=json.loads(raw)
                except (TypeError,json.JSONDecodeError): return '<MISSING>'
                v=parsed.get(primary_field,'<MISSING>')
                if pd.isna(v): return '<NA>'
                if isinstance(v,(bool,np.bool_)): return str(bool(v)).lower()
                if isinstance(v,(int,float,np.integer,np.floating)): return f'{float(v):.12g}'
                return str(v).strip()
            value_overall=(m.loc[m.status.eq('PASS')]
                .assign(_primary_value=lambda d:d.screen_parameters.map(field_value))
                .groupby('_primary_value').localization_distance_mm.median().to_dict())
        stats=[]
        for cfg,g in m.groupby('configuration_key',sort=False):
            v=g.loc[g.status.eq('PASS'),'localization_distance_mm'].dropna().to_numpy(float)
            pv=field_value(g.screen_parameters.iloc[0]) if primary_field else None
            stats.append({'key':cfg,'median':np.median(v) if len(v) else np.inf,
                          'mean':np.mean(v) if len(v) else np.inf,
                          'pass':len(v),
                          'parameter_field':primary_field,
                          'parameter_value':pv,
                          'parameter_value_overall_median':value_overall.get(pv,np.inf)})
        ranked=sorted(stats,key=lambda v:(v['median'],-v['pass'],
            v['parameter_value_overall_median'],v['mean'],str(v['key'])))
        best=ranked[0]
        if not np.isfinite(best['median']):
            raise ValueError(f'{mode}: {method} has no successful localization in eligible grid')
        chosen=m[m.configuration_key.eq(best['key'])].copy()
        chosen['localization_error_mm']=chosen['localization_distance_mm'].where(chosen.status.eq('PASS'))
        chosen['view']='selected_configuration'
        selected.append(chosen)
        defs.append({'method':method,'configuration_key':best['key'],
            'configuration_parameters':chosen.configuration_parameters.iloc[0],
            'candidate_configurations_searched':len(stats),
            'selected_median_mm':best['median'],
            'primary_parameter':best['parameter_field'],
            'primary_parameter_value':best['parameter_value'],
            'primary_parameter_overall_median_mm':best['parameter_value_overall_median'],
            'selected_pass':best['pass'],'selected_fail':len(ref)-best['pass'],
            'selection_label':'lowest configuration median; ties use primary-parameter overall median, then configuration mean (in sample)'})
    chosen=pd.concat(selected,ignore_index=True)
    return chosen,pd.DataFrame(defs)


def make_report(mod, grid, audit, ref, root, args, mode,
                parameter_audit=None, configuration_screen=None):
    import matplotlib.pyplot as plt
    destination=root/mode
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f'{destination} exists; pass --overwrite')
    # Never remove existing outputs; overwrite individual named files only.
    tables=destination/'tables'; figures=destination/'figures'
    tables.mkdir(parents=True,exist_ok=True); figures.mkdir(parents=True,exist_ok=True)
    selected, definitions=select_and_overall(grid,METHODS,ref,mode)
    definitions['report_mode']=mode
    selected_all=selected[['subject','run','method','status','localization_error_mm','configuration_key']].copy()
    selected_pass=selected_all[selected_all.status.eq('PASS')].copy()
    selected_pass['view']='selected_configuration'
    # Keep failure as missing value for plotting and pairwise statistics.
    selected_plot=selected_all[['subject','run','method','localization_error_mm']].copy()
    wide=mod.to_wide(selected_plot)
    overall=mod.create_overall_run_values(grid.iloc[:0],grid)
    overall_pass=overall.dropna(subset=['localization_error_mm']).copy()
    selected_summary=mod.method_summary(selected_pass)
    overall_summary=mod.method_summary(overall_pass)
    compare=mod.selected_vs_overall_summary(selected_summary,overall_summary)
    # Merge reliability information explicitly, without treating failed errors as zero.
    compare=compare.merge(definitions[['method','selected_pass','selected_fail']],on='method')
    compare['selected_success_rate']=compare.selected_pass/len(ref)
    pair=mod.pairwise_summary(wide)
    overall_wide=mod.to_wide(overall)
    overall_pair=mod.pairwise_summary(overall_wide)
    participant=mod.participant_summary(selected_pass)
    overall_participant=mod.participant_summary(overall_pass)
    run=mod.run_summary(wide)
    overall_run=mod.run_summary(overall_wide)
    lowest=mod.lowest_counts(run)
    overall_lowest=mod.lowest_counts(overall_run)
    outliers=mod.outlier_table(selected_pass,1.5)
    overall_outliers=mod.outlier_table(overall_pass,1.5)
    overview=pd.DataFrame([{'report_mode':mode,'participants':len({x[0] for x in ref}),
        'matched_runs':len(ref),'methods':len(METHODS),'eligible_configurations':grid[['method','configuration_key']].drop_duplicates().shape[0],
        'attempted_grid_solutions_all_methods':len(grid),
        'pass_grid_solutions_all_methods':int(grid.status.eq('PASS').sum()),
        'fail_grid_solutions_all_methods':int(grid.status.eq('FAIL').sum())}])
    outputs={
      '01_selected_configuration_definitions.csv':definitions,
      '02_selected_configuration_method_summary.csv':selected_summary,
      '03_overall_grid_method_summary.csv':overall_summary,
      '04_selected_vs_overall_summary.csv':compare,
      '05_selected_configuration_pairwise_differences.csv':pair,
      '06_overall_grid_pairwise_differences.csv':overall_pair,
      '07_selected_configuration_participant_summary.csv':participant,
      '08_overall_grid_participant_summary.csv':overall_participant,
      '09_selected_configuration_run_summary.csv':run,
      '10_overall_grid_run_summary.csv':overall_run,
      '11_selected_configuration_lowest_error_counts.csv':lowest,
      '12_overall_grid_lowest_typical_counts.csv':overall_lowest,
      '13_selected_configuration_results_long.csv':selected_all,
      '14_selected_configuration_results_wide.csv':wide.reset_index(),
      '15_overall_grid_run_values_long.csv':overall,
      '16_overall_grid_run_values_wide.csv':overall_wide.reset_index(),
      '17_selected_configuration_outliers.csv':outliers,
      '18_overall_grid_outliers.csv':overall_outliers,
      '19_dataset_summary.csv':overview,
      '20_configuration_completeness_audit.csv':audit,
      '21_grid_eligible_rows_by_method.csv':grid.groupby(['method','status']).size().rename('rows').reset_index(),
      '22_parameter_value_reliability.csv':parameter_audit if parameter_audit is not None else pd.DataFrame(),
      '23_configuration_parameter_screen.csv':configuration_screen if configuration_screen is not None else pd.DataFrame(),
    }
    for name,df in outputs.items():df.to_csv(tables/name,index=False)
    if not args.no_figures:
        # All Script-17 plotting functions call Figure.savefig. Decorating that
        # call updates visible labels consistently in every PNG, without
        # altering METHOD_ORDER, merges, pivots, or analysis tables.
        from matplotlib.figure import Figure
        original_savefig = Figure.savefig
        def savefig_with_years(fig, *a, **kw):
            add_year_labels_to_figure(fig)
            return original_savefig(fig, *a, **kw)
        Figure.savefig = savefig_with_years
        mod.setup_plot_style()
        context=f'{len(ref)} runs | {len({x[0] for x in ref})} participants | 12 methods | {mode}'
        save_grouped_distribution(selected_pass,figures/'01_selected_configuration_distribution.png',
            'Localization error for selected lowest-median observed configurations',context,
            'Localization error (mm)',args.dpi)
        save_grouped_distribution(overall_pass,figures/'02_overall_grid_run_distribution.png',
            'Typical localization error across tested parameter grids',context,
            'Within-run median localization error (mm)',args.dpi)
        mod.save_selected_vs_overall(compare,figures/'03_selected_vs_overall_method_medians.png',args.dpi)
        mod.save_pairwise_heatmap(pair,figures/'04_selected_pairwise_median_differences.png',args.dpi)
        mod.save_participant(participant,figures/'05_selected_configuration_by_participant.png',
            'Selected fixed configurations: median error by participant',args.dpi)
        mod.save_lowest_counts(lowest,figures/'06_selected_configuration_lowest_error_counts.png',args.dpi)
        mod.save_participant(overall_participant,figures/'07_overall_grid_by_participant.png',
            'Typical within-grid run values by participant',args.dpi)
        mod.save_run_difficulty(run,figures/'08_selected_run_difficulty.png',args.dpi)
        mod.save_methods_by_run(wide,figures/'09_selected_configuration_by_run.png',
            'Selected fixed configurations: localization error by run',context,
            'Localization error (mm)',args.dpi)
        mod.save_methods_by_run(overall_wide,figures/'10_overall_grid_by_run.png',
            'Overall parameter grids: within-run median localization error',context,
            'Within-run median localization error (mm)',args.dpi)
        plt.close('all')
    print(f'\n{mode}: {len(grid):,} grid rows; {len(definitions)} methods; '
          f'PASS={int(grid.status.eq("PASS").sum()):,}, FAIL={int(grid.status.eq("FAIL").sum()):,}')
    print(compare[['method','selected_median_mm','selected_pass','selected_fail',
                   'overall_median_mm']].to_string(index=False))
    print(f'Wrote: {destination}')


def write_merged_result_files(root, all_results, all_pass, parameter_audit,
                              configuration_audit, overwrite=False):
    """Write the two merged method grids and their screening audit at output root."""
    root.mkdir(parents=True, exist_ok=True)
    outputs={
        'all_methods_complete_configuration_results.csv':all_results,
        'all_methods_all_pass_configurations.csv':all_pass,
        'all_methods_parameter_value_reliability.csv':parameter_audit,
        'all_methods_configuration_eligibility.csv':configuration_audit,
    }
    existing=[name for name in outputs if (root/name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Merged output(s) already exist in {root}: {existing}; pass --overwrite')
    for name,frame in outputs.items():
        frame.to_csv(root/name,index=False)
    print(f'Wrote merged results: {root}/all_methods_complete_configuration_results.csv '
          f'({len(all_results):,} rows)')
    print(f'Wrote strict all-pass results: {root}/all_methods_all_pass_configurations.csv '
          f'({len(all_pass):,} rows)')


def save_grouped_distribution(long_table, path, title, subtitle, ylabel, dpi):
    """Boxplot styled to match Script 17's grouped-method figure."""
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(17,7.5))
    values=[long_table.loc[long_table.method.eq(m),'localization_error_mm']
            .dropna().to_numpy(float) for m in METHODS]
    positions=np.arange(1,len(METHODS)+1)
    boxes=ax.boxplot(values,positions=positions,patch_artist=True,showfliers=True,
        widths=.58,medianprops={'color':'#222222','linewidth':1.4},
        whiskerprops={'color':'#555555','linewidth':1.0},
        capprops={'color':'#555555','linewidth':1.0},
        flierprops={'marker':'o','markerfacecolor':'white','markeredgecolor':'#333333',
                    'markersize':4.5,'linestyle':'none'})
    for patch,method in zip(boxes['boxes'],METHODS):
        patch.set_facecolor(COLORS[method]); patch.set_alpha(.82)
        patch.set_edgecolor('#555555'); patch.set_linewidth(.9)
    labels=[f'{m}\n({METHOD_YEARS[m]})' for m in METHODS]
    ax.set_xticks(positions,labels,rotation=35,ha='right',rotation_mode='anchor')
    ax.set_xlim(.5,len(METHODS)+.5)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    ax.set_title(f'{title}\n{subtitle}',pad=14)
    ax.grid(axis='y',alpha=.25)
    start=1
    for group_name,group_methods in METHOD_GROUPS:
        end=start+len(group_methods)-1
        center=(start+end)/2
        ax.text(center,-.43,group_name,transform=ax.get_xaxis_transform(),
                ha='center',va='top',fontsize=10,fontweight='bold',clip_on=False)
        if end<len(METHODS):
            ax.axvline(end+.5,color='#bdbdbd',linestyle=':',linewidth=1.2,zorder=0)
        start=end+1
    fig.subplots_adjust(left=.07,right=.99,top=.88,bottom=.39)
    fig.savefig(path,dpi=dpi,bbox_inches='tight')
    plt.close(fig)


def main():
    args=arguments()
    mod=script17_module(args.script17)
    old=old_grid(args,mod)
    old=prepare_old(old,reference_keys(old.loc[old.method.eq('MNE')]))
    ref=reference_keys(old.loc[old.method.eq('MNE')])
    if len(ref)!=args.expected_runs:
        raise ValueError(f'Original MNE reference contains {len(ref)} runs, expected {args.expected_runs}')
    old_complete,audit_complete=eligibility(old,ref,False)
    args.output_root.mkdir(parents=True,exist_ok=True)
    completeness_path=args.output_root/'input_configuration_completeness.csv'
    if completeness_path.exists() and not args.overwrite:
        raise FileExistsError(f'{completeness_path} exists; pass --overwrite')
    audit_complete.to_csv(completeness_path,index=False)
    print('Input grid completeness (before parameter-wide PASS filtering):')
    print(audit_complete.groupby('method').agg(
        configurations=('configuration_key','size'),
        largest_runs_per_configuration=('runs','max'),
        complete_configurations=('eligible','sum')).to_string())
    incoming_any=load_new(args.new_any,NEW,ref,mod)
    incoming_pass=load_new(args.new_pass,NEW,ref,mod)
    if not incoming_pass.status.eq('PASS').all():
        raise ValueError('Script 25 all-pass dataset contains failures')
    expected_pass_keys=set(incoming_any.loc[
        incoming_any.groupby(['method','configuration_key']).status.transform(
            lambda s:s.eq('PASS').all()),'configuration_key'])
    observed_pass_keys=set(incoming_pass.configuration_key)
    if expected_pass_keys!=observed_pass_keys:
        raise ValueError('Script 25 06/07 datasets disagree on all-pass configurations')
    # Merge Script 17 methods + MxNE alpha extensions + Script 25 next methods.
    # The all-results file retains every complete configuration, including FAIL
    # rows. The strict file retains only configurations whose full parameter
    # combination passes all runs and whose primary parameter value has no
    # failures anywhere across the merged grid.
    merged_any=pd.concat([old_complete,incoming_any],ignore_index=True,sort=False)
    raw_new=load_new_reliability_evidence(args.new_master,mod)
    evidence=pd.concat([old,raw_new],ignore_index=True,sort=False)
    screened,parameter_audit,configuration_screen=parameter_value_screen(merged_any,evidence,ref)
    merged_all_pass,audit_all_pass=eligibility(screened,ref,True)
    if not merged_all_pass.status.eq('PASS').all():
        raise ValueError('Strict all-pass screen retained a failed observation')
    expected_methods=set(METHODS)
    missing_any=sorted(expected_methods-set(merged_any.method))
    unexpected_any=sorted(set(merged_any.method)-expected_methods)
    if missing_any or unexpected_any:
        raise ValueError(f'Combined complete-results grid method mismatch; '
                         f'missing={missing_any}, unexpected={unexpected_any}')
    merged_config_audit=configuration_screen.merge(
        audit_all_pass[['method','configuration_key','runs','pass','fail','eligible']],
        on=['method','configuration_key'],how='left')
    merged_config_audit['eligible_all_runs_pass']=merged_config_audit.eligible.fillna(False)
    merged_config_audit=merged_config_audit.drop(columns=['eligible'])
    # Persist the combined tables before strict cohort validation, so a method
    # with no qualifying configuration still leaves a reviewable audit trail.
    write_merged_result_files(args.output_root,merged_any,merged_all_pass,
                              parameter_audit,merged_config_audit,args.overwrite)
    missing_all_pass=sorted(expected_methods-set(merged_all_pass.method))
    if missing_all_pass:
        config_counts=(merged_any.groupby('method').configuration_key.nunique()
                       .reindex(METHODS,fill_value=0).to_dict())
        passing_counts=(merged_all_pass.groupby('method').configuration_key.nunique()
                        .reindex(METHODS,fill_value=0).to_dict())
        excluded_values=(parameter_audit.loc[~parameter_audit.eligible_value]
                         .groupby('method').apply(
                             lambda d:'; '.join(f"{r.parameter}={r.value} (FAIL={r.fail_rows})"
                                                for r in d.itertuples()),
                             include_groups=False).to_dict())
        raise ValueError('No strict all-pass configuration remains for method(s): '
                         f'{missing_all_pass}. Complete configurations by method: {config_counts}. '
                         f'Strict all-pass configurations by method: {passing_counts}. '
                         f'Primary parameter values excluded for failures or incomplete coverage: {excluded_values}. '
                         'Combined CSVs and eligibility audits were written to --output-root.')

    for mode in ['complete_any_status','complete_all_pass']:
        configuration_screen_mode=configuration_screen
        parameter_audit_mode=parameter_audit
        if mode=='complete_any_status':
            all_data=merged_any
            audit_old=audit_complete
        else:
            all_data=merged_all_pass
            audit_old=audit_all_pass
        if set(all_data.method)!=set(METHODS):
            raise ValueError(f'Incomplete method set: {set(METHODS)-set(all_data.method)}')
        for meth,group in all_data.groupby('method'):
            if reference_keys(group)!=ref:
                raise ValueError(f'{mode}: {meth} run key mismatch')
        make_report(mod,all_data,audit_old,ref,args.output_root,args,mode,
                    parameter_audit_mode,configuration_screen_mode)

if __name__=='__main__':main()
