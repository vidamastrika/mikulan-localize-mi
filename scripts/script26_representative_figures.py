"""Figure-only Script 26 extension: fixed selected configurations, no inverse fits.

Reads the saved cohort selection, full grid coordinates, selected-run errors and
Script 07 anatomical geometry. Checks all 12 methods per case, coordinate units,
midpoint agreement and agreement of reconstructed Euclidean distances with the
saved error tables. Uses the original Script 07 overlay renderer. Outputs a new
representative_runs/ directory inside the selected Script 26 cohort.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()


def render(args, methods, colors, styles, default_cases):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    root=args.output_root.expanduser().resolve()
    cohort=root/args.report_mode
    output=cohort/'representative_runs'
    require(not output.exists() or args.overwrite,f'{output} exists; use --overwrite')
    cases=list(dict.fromkeys(tuple(c) for c in (args.representative_case or default_cases)))
    definitions_path=cohort/'tables/01_selected_configuration_definitions.csv'
    results_path=cohort/'tables/13_selected_configuration_results_long.csv'
    geometry_path=args.anatomical_geometry.expanduser().resolve()
    grid_path=root/('all_methods_all_pass_configurations.csv' if args.report_mode=='complete_all_pass'
                   else 'all_methods_complete_configuration_results.csv')
    definitions=pd.read_csv(definitions_path)
    require(len(definitions)==12 and set(definitions.method)==set(methods)
            and definitions.method.is_unique,'Expected one selected configuration per method')
    keys=dict(zip(definitions.method,definitions.configuration_key))
    wanted={(s,r,m) for s,r in cases for m in methods}
    available=pd.read_csv(grid_path,nrows=0).columns
    columns=['subject','run','method','status','configuration_key','configuration_parameters',
             'montage','channels','stimulation_pair','hemisphere','peak_vertex','peak_time_ms','selected_time_ms',
             'localization_distance_mm','selection_criterion']
    for axis in 'xyz':
        columns += [f'estimated_{axis}_m',f'selected_{axis}_mm',f'known_{axis}_m',f'ground_truth_{axis}_mm']
    parts=[]
    print('Reading saved grid coordinates for selected configurations (no inverses)...',flush=True)
    for chunk in pd.read_csv(grid_path,usecols=[c for c in columns if c in available],
                             chunksize=30000,low_memory=False):
        mask=chunk.configuration_key.eq(chunk.method.map(keys))
        mask &= pd.Series(list(zip(chunk.subject,chunk.run))).isin(cases).to_numpy()
        if mask.any():parts.append(chunk.loc[mask].copy())
    require(parts,'No saved selected-configuration coordinates found')
    chosen=pd.concat(parts,ignore_index=True)
    observed=set(map(tuple,chosen[['subject','run','method']].to_numpy()))
    require(len(chosen)==len(wanted) and observed==wanted,'Missing/duplicate selected case-method rows')
    require(chosen.status.eq('PASS').all(),'Selected cohort has failed case-method rows; no missing estimates are fabricated')
    sources=[]
    for i,row in chosen.iterrows():
        used=[]
        for axis in 'xyz':
            for destination,candidates in [
                (f'estimated_{axis}_m',[(f'estimated_{axis}_m',1.),(f'selected_{axis}_mm',.001)]),
                (f'known_{axis}_m',[(f'known_{axis}_m',1.),(f'ground_truth_{axis}_mm',.001)]),
            ]:
                value=None
                for name,scale in candidates:
                    raw=pd.to_numeric(pd.Series([row.get(name)]),errors='coerce').iloc[0]
                    if np.isfinite(raw):
                        value=float(raw)*scale;used.append(f'{name}*{scale:g}')
                        break
                require(value is not None,f'Missing coordinate: {row.method} {row.subject} {row.run} {destination}')
                chosen.loc[i,destination]=value
        sources.append(';'.join(used))
    chosen['coordinate_source_columns']=sources
    chosen['coordinate_frame']='individual_native_surface'
    chosen['coordinate_units']='metres'
    predicted=chosen[[f'estimated_{a}_m' for a in 'xyz']].to_numpy(float)
    known=chosen[[f'known_{a}_m' for a in 'xyz']].to_numpy(float)
    recomputed=np.linalg.norm(predicted-known,axis=1)*1000
    errors=pd.to_numeric(chosen.localization_distance_mm).to_numpy(float)
    require(np.isfinite(errors).all() and np.allclose(recomputed,errors,rtol=0,atol=1e-5),
            'Grid coordinates do not reproduce saved errors; check coordinate frames/units')
    chosen['coordinate_error_check_mm']=np.abs(recomputed-errors)
    reference=pd.read_csv(results_path)
    matched=chosen.merge(reference[['subject','run','method','configuration_key','localization_error_mm']],
                         on=['subject','run','method','configuration_key'],how='left',validate='one_to_one')
    require(matched.localization_error_mm.notna().all() and np.allclose(
            matched.localization_error_mm,matched.localization_distance_mm,rtol=0,atol=1e-6),
            'Representative results disagree with Script 26 selected results')
    run_geometry=pd.read_csv(geometry_path)
    run_geometry=run_geometry.loc[pd.Series(list(zip(run_geometry.subject,run_geometry.run))).isin(cases).to_numpy()].copy()
    require(len(run_geometry)==len(cases) and not run_geometry.duplicated(['subject','run']).any(),
            'Script 07 anatomical geometry does not cover the requested cases')
    script07=Path(__file__).with_name('07_inspect_anatomical_locations.py')
    spec=importlib.util.spec_from_file_location('script26_anatomical_renderer',script07)
    geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)
    geometry.METHOD_ORDER=list(methods)
    geometry.METHOD_COLORS=dict(colors)
    geometry.METHOD_MARKERS={m:styles[m][0] for m in methods}
    geometry.METHOD_LINESTYLES={m:styles[m][1] for m in methods}
    geometry.LEGEND_ANCHOR=(1.12,.015)
    geometry.parameter_context=lambda table,include_channel_count=False: (
        f'12 methods | fixed per-method best parameters | {args.report_mode}\n'
        'Selected across 61 runs (in sample); parameter details in accompanying CSV')
    dataset=args.dataset.expanduser().resolve()
    resources=[definitions_path,results_path,geometry_path,grid_path,script07,Path(__file__),
               Path(__file__).with_name('26_compare_twelve_methods.py')]
    for subject,run in cases:
        row=run_geometry.loc[run_geometry.subject.eq(subject)&run_geometry.run.eq(run)].iloc[0]
        native=row[[f'surface_{a}_m' for a in 'xyz']].to_numpy(float)
        group=chosen.loc[chosen.subject.eq(subject)&chosen.run.eq(run)]
        require(np.allclose(group[[f'known_{a}_m' for a in 'xyz']],native,rtol=0,atol=5e-5),
                f'Script 07 midpoint differs from selected coordinates: {subject} {run}')
        contacts=geometry.electrode_file(dataset,subject,'surface')
        point=geometry.contact_midpoint(contacts,[row.contact_1,row.contact_2])
        require(np.linalg.norm(point-native)*1000<=.05,'Released contact midpoint differs from Script 07')
        require(set(group.stimulation_pair)=={row.stimulation_pair},'Stimulation pair differs')
        resources.append(contacts)
        anat=dataset/'derivatives/sourcemodelling'/subject/'anat'
        resources += [anat/f'{subject}_hemi-{h}_pial.surf.gii' for h in ['L','R']]
        resources.append(anat/f'{subject}_outer_skin.surf.gii')
    # Validate every input before creating outputs. Preserve the existing
    # Script 26 tables, original Script 07 reports and active other experiment.
    print('Hashing figure provenance inputs...',flush=True)
    input_hashes={str(p.resolve()):digest(p) for p in dict.fromkeys(resources)}
    (output/'figures').mkdir(parents=True,exist_ok=True)
    chosen['method']=pd.Categorical(chosen.method,categories=methods,ordered=True)
    chosen=chosen.sort_values(['subject','run','method'])
    chosen.to_csv(output/'selected_run_method_coordinates.csv',index=False)
    definitions.to_csv(output/'selected_configuration_definitions.csv',index=False)
    run_geometry.to_csv(output/'run_anatomical_geometry.csv',index=False)
    pd.DataFrame([dict(method=m,color=colors[m],marker=styles[m][0],linestyle=styles[m][1])
                  for m in methods]).to_csv(output/'method_styles.csv',index=False)
    figures=[]
    for subject,run in cases:
        filename=output/'figures'/f'anatomical_geometry_{subject}_{run}.png'
        print(f'Rendering Script 07 overlay: {subject} {run} (12 methods)',flush=True)
        geometry.save_run_geometry_figure(dataset,run_geometry,chosen,subject,run,filename,args.dpi)
        figures.append(filename)
        group=chosen.loc[chosen.subject.eq(subject)&chosen.run.eq(run)].set_index('method').loc[methods]
        fig,ax=plt.subplots(figsize=(15,7))
        values=group.localization_distance_mm.to_numpy(float)
        bars=ax.bar(np.arange(12),values,color=[colors[m] for m in methods],alpha=.82)
        ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=3,fontsize=9)
        ax.set_xticks(np.arange(12),methods,rotation=35,ha='right')
        ax.set_ylabel('Localization error (mm)')
        ax.set_title(f'{subject} {run}: fixed selected configurations, all 12 methods\n'
                     f'{args.report_mode} | best median across 61 runs (in sample)')
        ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True);ax.set_ylim(0,max(values)*1.18)
        fig.tight_layout()
        filename=output/'figures'/f'localization_errors_{subject}_{run}.png'
        fig.savefig(filename,dpi=args.dpi,bbox_inches='tight');plt.close(fig)
        figures.append(filename)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),report_mode=args.report_mode,
        cases=cases,methods=list(methods),n_results=len(chosen),dpi=args.dpi,
        selection='Existing Script 26 per-method best fixed configuration across 61 runs; no per-case optimization',
        inverse_reconstruction_executed=False,selection_recomputed=False,
        coordinate_frame='individual native surface',coordinate_units='metres',
        maximum_coordinate_error_check_mm=float(chosen.coordinate_error_check_mm.max()),
        input_sha256=input_hashes,figures=[str(p.relative_to(output)) for p in figures],
        figure_sha256={str(p.relative_to(output)):digest(p) for p in figures},
        limitations=['In-sample descriptive selection, not held-out validation.',
                     'Coincident method estimates overlap in 3D; error bars and coordinate table retain all methods.',
                     'Anatomical reference geometry is reused from Script 07, with midpoint agreement checked.',
                     'Native-surface coordinates are used; no estimated MNI positions or atlas labels are invented.'])
    (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'README.md').write_text('# Script 26 representative-run figures\n\n'
        'All 12 methods use the saved cohort best fixed configuration across all 61 runs.\n'
        'These are in-sample selections, not per-run best parameters or held-out validation.\n\n'
        'figures/anatomical_geometry_*.png: Script 07-style pial/scalp overlay, stimulation,\n'
        'nearest source/pial/scalp/good sensor references and 12 method estimates.\n'
        'figures/localization_errors_*.png: matching per-method error bars.\n'
        'selected_run_method_coordinates.csv: selected keys, settings, coordinates and error checks.\n'
        'selected_configuration_definitions.csv: exact saved selection definitions.\n'
        'run_anatomical_geometry.csv: reused geometry for the pictured runs.\n'
        'method_styles.csv: shared Script 26 colors and markers/line styles.\n'
        'manifest.json: SHA-256 provenance, validation, cases and interpretation.\n\n'
        'No inverse was rerun; no original analysis table or Script 07 output was changed.\n'
        'Coincident estimates can overlap in 3D; the tables and error bars retain all 12 results.\n')
    print(f'COMPLETE: {len(figures)} figures, {len(chosen)} selected results. {output}',flush=True)
