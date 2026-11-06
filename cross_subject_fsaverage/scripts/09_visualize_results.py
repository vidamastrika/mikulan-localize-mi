#!/usr/bin/env python3
"""Create fsaverage diagnostic PNGs and Markdown/HTML reports in no-warp styles.

CSV-only analysis; no inverse reconstruction or MRI registration. Existing output
folders are refused. All-seven, other-six and sub-03 cohorts are reported.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import html
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'cross_subject_fsaverage'
STYLE = ROOT / 'cross_subject_no_warp/scripts/09_visualize_results.py'
spec = importlib.util.spec_from_file_location('no_warp_plot_style', STYLE)
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)
METHODS = style.METHOD_ORDER
SUBJECTS = [f'sub-{i:02}' for i in range(1, 8)]
KEY = ['strategy', 'method', 'test_subject', 'reference_subject', 'run']
COHORTS = {
    'all_subjects_including_sub03': ('All seven subjects (includes exploratory sub-03)', SUBJECTS),
    'six_subjects': ('Six subjects (anatomical review required)', [s for s in SUBJECTS if s != 'sub-03']),
    'sub03_exploratory': ('Sub-03 only (failed alignment QA; exploratory)', ['sub-03']),
}
COLORS = {'Historical baseline': '#4E79A7', 'LOSO': '#59A14F', 'Single-reference': '#E15759'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def boxplot(series, path, title, colored_methods=False):
    fig, ax = plt.subplots(figsize=(19.5, 10))
    centers = np.arange(1, 13)
    offsets = np.linspace(-.26, .26, len(series)) if len(series) > 1 else [0]
    width = min(.32, .65 / len(series))
    palette = plt.get_cmap('tab20')([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 5])
    for j, (label, frame, color) in enumerate(series):
        for i, method in enumerate(METHODS):
            vals = frame.loc[frame.method.eq(method), 'value_mm'].dropna().to_numpy(float)
            if not len(vals):
                continue
            bp = ax.boxplot([vals], positions=[centers[i] + offsets[j]], widths=width,
                patch_artist=True, showfliers=colored_methods, manage_ticks=False,
                flierprops=dict(marker='.', markersize=2.5, alpha=.3))
            style.style_boxplot(bp, palette[i] if colored_methods else color)
            if colored_methods:
                bp['boxes'][0].set(hatch=['', '//', 'xx'][j % 3], alpha=[.95, .65, .40][j % 3])
    style.add_group_spans_and_labels(ax)
    ax.set_xticks(centers, [style.method_label(m) for m in METHODS], rotation=35, ha='right')
    ax.set(xlim=(.4, 12.6), ylabel='Distance (mm)')
    fig.suptitle(title, fontsize=16, weight='bold', y=.98)
    ax.grid(axis='y', alpha=.25)
    handles = [Patch(facecolor='#7f8790' if colored_methods else color,
                     hatch=['', '//', 'xx'][j % 3] if colored_methods else '',
                     alpha=[.95, .65, .40][j % 3] if colored_methods else .75, label=label)
               for j, (label, _, color) in enumerate(series)]
    ax.legend(handles=handles, loc='upper right', ncol=min(len(series), 3))
    fig.text(.5, .025, 'Fsaverage/no-warp: diagnostic HEAD separation. Historical baseline: within-subject error (in-sample selection).',
             ha='center', fontsize=10)
    fig.subplots_adjust(bottom=.23, top=.83, left=.07, right=.98)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def heatmap(frame, rows, cols, path, title):
    array = frame.reindex(index=rows, columns=cols).to_numpy(float)
    fig, ax = plt.subplots(figsize=(max(9, len(cols) * 1.05), max(4.8, len(rows) * .65 + 2)))
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('#eeeeee')
    im = ax.imshow(np.ma.masked_invalid(array), aspect='auto', cmap=cmap)
    ax.set_xticks(np.arange(len(cols)), cols, rotation=40, ha='right')
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_title(title, fontsize=13, weight='bold', pad=15)
    finite = array[np.isfinite(array)]
    midpoint = (finite.min() + finite.max()) / 2 if finite.size else 0
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = array[i, j]
            ax.text(j, i, f'{v:.1f}' if np.isfinite(v) else '—', ha='center', va='center',
                    fontsize=8, color=('black' if v > midpoint else 'white') if np.isfinite(v) else '#777777')
    fig.colorbar(im, ax=ax, label='Diagnostic distance (mm)')
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def stats(frame, groups):
    return frame.groupby(groups, dropna=False).agg(
        n_total=('status', 'size'), n_pass=('status', lambda s: int(s.eq('PASS').sum())),
        n_values=('value_mm', 'count'), median_mm=('value_mm', 'median'),
        q25_mm=('value_mm', lambda s: s.quantile(.25)), q75_mm=('value_mm', lambda s: s.quantile(.75)),
        mean_mm=('value_mm', 'mean')).reset_index().assign(
            pass_rate=lambda d: d.n_pass / d.n_total)


def cohort_plots(frame, baseline, paired, out, label):
    out.mkdir()
    master_statistics = []
    for strategy, sample in [('Baseline', baseline), ('LOSO', frame[frame.strategy.eq('LOSO')]),
                             ('Single-reference', frame[frame.strategy.eq('Single-reference')])]:
        for method, g in sample.groupby('method'):
            values = g.value_mm.dropna()
            master_statistics.append(dict(method=method, strategy=strategy, N=len(values),
                Mean_mm=values.mean(), Median_mm=values.median(), SD_mm=values.std(),
                Q1_mm=values.quantile(.25), Q3_mm=values.quantile(.75),
                IQR_mm=values.quantile(.75)-values.quantile(.25), Min_mm=values.min(), Max_mm=values.max()))
    pd.DataFrame(master_statistics).sort_values(['method','strategy']).round(3).to_csv(
        out/'00_master_three_strategies_statistics.csv', index=False)
    series = [(s, frame[frame.strategy.eq(s)], COLORS[s]) for s in ['LOSO', 'Single-reference']]
    boxplot([('Historical baseline', baseline, COLORS['Historical baseline'])] + series,
            out/'00_master_three_strategies_boxplot.png', 'Baseline and fsaverage strategies\n' + label, True)
    boxplot([('Historical baseline', baseline, COLORS['Historical baseline'])],
            out/'01_historical_baseline_boxplot.png', 'Historical within-subject baseline context\n' + label)
    boxplot(series, out/'02_cross_boxplot.png', 'Fsaverage diagnostic HEAD separation\n' + label)
    fig, ax = plt.subplots(figsize=(18, 8))
    x = np.arange(1, 13)
    for j, (strategy, g, color) in enumerate(series):
        rates = g.groupby('method').status.apply(lambda s: s.eq('PASS').mean()*100).reindex(METHODS)
        ax.bar(x + (j-.5)*.34, rates, width=.34, color=color, label=strategy)
    style.add_group_spans_and_labels(ax)
    ax.set_xticks(x, [style.method_label(m) for m in METHODS], rotation=40, ha='right')
    ax.set(ylabel='Solver pass rate (%)', ylim=(0, 105))
    fig.suptitle('Fsaverage solver completion\n' + label, y=.98, weight='bold')
    ax.legend(); ax.grid(axis='y', alpha=.25)
    fig.subplots_adjust(bottom=.23, top=.80)
    fig.savefig(out/'03_pass_rate_grouped.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    sr = frame[frame.strategy.eq('Single-reference')]
    sens = sr.groupby(['method', 'test_subject', 'run']).agg(
        n_references=('value_mm', 'count'), sd_mm=('value_mm', 'std'),
        range_mm=('value_mm', lambda s: s.max()-s.min())).reset_index()
    # Require all six successful references when describing six-reference variability.
    complete = sens[sens.n_references.eq(6)]
    ss = complete.groupby('method')[['sd_mm', 'range_mm']].median().reindex(METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(19, 8))
    for ax, metric, name, color in zip(axes, ['sd_mm','range_mm'], ['SD','Range'], ['#4E79A7','#E15759']):
        ax.bar(x, ss[metric], color=color)
        style.add_group_spans_and_labels(ax)
        ax.set_xticks(x, [style.method_label(m) for m in METHODS], rotation=40, ha='right')
        ax.set_ylabel(f'{name} (mm)')
        ax.set_title(f'Median {name} across six references', y=1.18, pad=12)
        for text in ax.texts:
            text.set_fontsize(8)
            text.set_text(text.get_text().replace('Dipole / Beamformer', 'Dipole /\nBeamformer'))
        ax.grid(axis='y', alpha=.25)
    fig.suptitle('Reference sensitivity: complete six-reference test runs\n' + label, y=1.02)
    fig.subplots_adjust(bottom=.24, top=.70, wspace=.18)
    fig.savefig(out/'04_reference_sensitivity.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    refs = sr.groupby(['reference_subject','method']).value_mm.median().unstack('method')
    heatmap(refs, SUBJECTS, METHODS, out/'05_reference_by_method_heatmap.png', 'Median fsaverage diagnostic distance by reference\n' + label)
    pairs = sr.groupby(['test_subject','reference_subject','method']).value_mm.median().reset_index()
    matrix = pairs.groupby(['test_subject','reference_subject']).value_mm.median().unstack('reference_subject')
    targets = sorted(frame.test_subject.unique())
    heatmap(matrix, targets, SUBJECTS, out/'06_test_reference_heatmap.png',
            'Median of method-specific median distances by test/reference\n' + label)
    for strategy in ['LOSO', 'Single-reference']:
        matrix = frame[frame.strategy.eq(strategy)].groupby(['test_subject','method']).value_mm.median().unstack('method')
        heatmap(matrix, targets, METHODS, out/f'07_{strategy.lower()}_subject_method_heatmap.png',
                f'{strategy}: median diagnostic distance by test subject\n' + label)
    if len(paired):
        compare = []
        for strategy in ['LOSO','Single-reference']:
            g = paired[paired.strategy.eq(strategy)]
            for column, color in [('no_warp_mm','#4E79A7'), ('fsaverage_mm',COLORS[strategy])]:
                compare.append((f'{strategy}: '+('no-warp primary' if column=='no_warp_mm' else 'fsaverage'),
                                g.rename(columns={column:'value_mm'}), color))
        boxplot(compare, out/'08_no_warp_vs_fsaverage_boxplot.png',
                'Matched successful jobs: no-warp primary and fsaverage\n' + label)
    return sens


def table_text(df):
    columns = list(df.columns)
    def fmt(v):
        if pd.isna(v): return '—'
        if isinstance(v, (float, np.floating)): return f'{v:.2f}'
        return str(v)
    cells = [[fmt(v) for v in row] for row in df.itertuples(index=False, name=None)]
    md = '| ' + ' | '.join(columns) + ' |\n| ' + ' | '.join(['---']*len(columns)) + ' |\n'
    md += '\n'.join('| '+' | '.join(row)+' |' for row in cells) + '\n'
    ht = '<table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr></thead><tbody>'
    ht += ''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in cells)+'</tbody></table>'
    return md, ht


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-root', type=Path, default=EXP/'outputs/07_evaluate_localization')
    p.add_argument('--no-warp-master', type=Path, default=ROOT/'cross_subject_no_warp/outputs/07_compare_strategies/01_master_run_level.csv')
    p.add_argument('--output-root', type=Path, default=EXP/'outputs/09_visualize_results')
    args = p.parse_args()
    out = args.output_root.resolve()
    require(not out.exists(), f'Choose a fresh output directory: {out}')
    source = args.evaluation_root.resolve()/'01_run_level_diagnostic.csv'
    manifest_path = args.evaluation_root.resolve()/'07_evaluation_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for path, expected in manifest['input_sha256'].items():
        require(sha(path)==expected, f'Evaluation input changed; regenerate evaluation: {path}')
    hashes = {str(path.resolve()): sha(path) for path in [source,manifest_path,args.no_warp_master,STYLE,Path(__file__)]}
    data = pd.read_csv(source, keep_default_na=False)
    data['value_mm'] = pd.to_numeric(data.diagnostic_distance_mm, errors='coerce')
    require(len(data)==5124 and not data.duplicated(KEY).any(), 'Expected exact 5,124-job ledger')
    require(data.status.isin(['PASS','FAIL']).all(), 'Finish reconstruction before creating the final report')
    require(set(data.method)==set(METHODS), 'Expected all 12 methods')
    require(data.loc[data.status.eq('PASS'),'value_mm'].notna().all(), 'Missing PASS distances')
    require(data.loc[data.status.eq('FAIL'),'value_mm'].isna().all(), 'FAIL rows must not contain distances')
    old = pd.read_csv(args.no_warp_master, keep_default_na=False)
    old['value_mm'] = pd.to_numeric(old.value_mm, errors='coerce')
    baseline = old[old.experiment.eq('Historical_within') & old.scenario.eq('all_pass_selected')].copy()
    require(len(baseline)==732 and not baseline.duplicated(['method','test_subject','run']).any(), 'Historical baseline must occur once per method/test run')
    require(baseline.status.eq('PASS').all(), 'Expected historical all-pass baseline')
    old_cross = old[old.scenario.eq('primary') & old.experiment.isin([
        'LOSO_single_reference_no_warp','Single-reference_single_reference_no_warp'])].copy()
    require(not old_cross.duplicated(KEY).any(), 'Duplicate no-warp primary jobs')
    require(np.isfinite(old_cross.loc[old_cross.status.eq('PASS'),'value_mm']).all(), 'Nonfinite no-warp PASS distances')
    paired = data[KEY+['status','value_mm']].merge(old_cross[KEY+['status','value_mm']],
        on=KEY, how='outer', suffixes=('_fsaverage','_no_warp'), indicator=True, validate='one_to_one')
    paired['jointly_pass'] = paired.status_fsaverage.eq('PASS') & paired.status_no_warp.eq('PASS')
    paired['fsaverage_mm'] = pd.to_numeric(paired.value_mm_fsaverage, errors='coerce')
    paired['no_warp_mm'] = pd.to_numeric(paired.value_mm_no_warp, errors='coerce')
    paired['paired_delta_mm'] = (paired.fsaverage_mm-paired.no_warp_mm).where(paired.jointly_pass)
    out.mkdir(parents=True); tables=out/'tables'; tables.mkdir()
    data.to_csv(tables/'01_fsaverage_run_level.csv', index=False)
    paired.to_csv(tables/'02_no_warp_matched_jobs.csv', index=False)
    summaries=[]; sensitivities=[]; deltas=[]
    md=['# Fsaverage reconstruction and diagnostic distance report\n']; ht=['<h1>Fsaverage reconstruction and diagnostic distance report</h1>']
    def paragraph(text):
        md.append(text+'\n'); ht.append('<p>'+html.escape(text)+'</p>')
    def heading(text):
        md.append('## '+text+'\n'); ht.append('<h2>'+html.escape(text)+'</h2>')
    def table(df):
        a,b=table_text(df); md.append(a); ht.append(b)
    paragraph(f'All {len(data):,} planned jobs have finished: {data.status.eq("PASS").sum():,} PASS and {data.status.eq("FAIL").sum()} FAIL. Sub-03 has {data.loc[data.test_subject.eq("sub-03"), "status"].eq("PASS").sum()} successful jobs. PASS means numerical solver completion; it does not establish anatomical accuracy.')
    paragraph('The measured quantity is unregistered Euclidean HEAD-coordinate separation in millimetres. The stimulation midpoint was converted from each test subject’s native surface coordinates to native HEAD using its own H5 transform. No MRI registration or intersubject anatomical mapping was applied. localization_error_valid remains false throughout.')
    paragraph('Sub-03 uses a candidate sensor-to-template alignment that failed QA and remains exploratory. The other six subjects retain anatomical review requirements. The report includes all seven and also supplies separate six-subject and sub-03 analyses.')
    heading('Experiment and analysis policy')
    paragraph('Twelve methods use fsaverage anatomy with test-specific forward models and frozen LOSO or single-reference parameter selections. Reference identities also control nominal channel selection and bad-channel replacement. Cortical methods retain the original known-stimulation-hemisphere peak policy; Continuous-ECD and RAP-MUSIC retain maximum-GOF selection. No whole-brain peak sensitivity results were generated for this fsaverage run.')
    paragraph('Historical baseline values are the original all-pass, in-sample-selected within-subject errors, included once per test run. Their metric differs from the cross-anatomy diagnostic distances. They provide context rather than an unbiased held-out performance benchmark. Baseline plots do not represent a new fsaverage within-subject fit.')
    paragraph('Boxplot samples are successful run-level results. Single-reference results repeat each test run under six reference identities; those observations are not independent participants. Failure rates retain all planned jobs in the denominator. Equal-subject summaries first calculate each test subject’s median across its available run/reference results, then average those medians. No statistical significance tests are claimed.')
    heading('Completion and failures')
    table(data.groupby(['strategy','status']).size().rename('count').reset_index())
    failures=data[data.status.eq('FAIL')][KEY+['failure_reason']]
    table(failures)
    paragraph('All six failures were inherited Gamma-MAP cases with no active dipoles at the frozen alpha. Their missing distances were not imputed, and none occurred in sub-03.')
    for cohort,(label,subjects) in COHORTS.items():
        frame=data[data.test_subject.isin(subjects)].copy()
        base=baseline[baseline.test_subject.isin(subjects)]
        pairs=paired[paired.test_subject.isin(subjects)]
        sens=cohort_plots(frame,base,pairs[pairs.jointly_pass],out/cohort,label)
        sens['cohort']=cohort; sensitivities.append(sens)
        summary=stats(frame,['strategy','method']); summary['cohort']=cohort; summaries.append(summary)
        subject_stats=stats(frame,['strategy','method','test_subject'])
        equal=subject_stats.groupby(['strategy','method']).median_mm.mean().rename('equal_subject_mean_of_medians_mm').reset_index()
        subject_stats.to_csv(tables/f'{cohort}_subject_summary.csv',index=False)
        equal.to_csv(tables/f'{cohort}_equal_subject_summary.csv',index=False)
        heading(label)
        paragraph(f'{len(subjects)} test subject(s), {len(frame)} jobs, {frame.status.eq("PASS").sum()} PASS. Historical baseline: {len(base)} distinct method/test-run values. Values below are pooled successful diagnostic distances; see equal-subject CSV for participant weighting.')
        display=summary[['strategy','method','n_total','n_pass','median_mm','q25_mm','q75_mm']].copy()
        table(display)
        for strategy in ['LOSO','Single-reference']:
            g=summary[summary.strategy.eq(strategy)].sort_values('median_mm').iloc[0]
            paragraph(f'{strategy}: smallest pooled median diagnostic distance is {g.method}, {g.median_mm:.2f} mm. This is descriptive and does not establish a validated anatomical accuracy ranking.')
        excluded=int(sens.n_references.ne(6).sum())
        paragraph(f'Reference sensitivity uses sample SD (ddof=1) and range across six successful references per method/test/run. {excluded} method/test/run groups lack six successful references and are excluded from these variability summaries, while remaining in completion tables. Heatmaps display available medians; a blank diagonal means a subject is not its own single reference.')
        valid=pairs[pairs.jointly_pass]
        ps=valid.groupby(['strategy','method','test_subject']).paired_delta_mm.median().reset_index()
        ds=ps.groupby(['strategy','method']).agg(n_subjects=('test_subject','nunique'),mean_subject_median_delta_mm=('paired_delta_mm','mean')).reset_index()
        ds['cohort']=cohort; deltas.append(ds)
        paragraph(f'Matched no-warp comparison: {len(valid)} jointly successful jobs of {len(pairs)} union keys. Negative paired deltas mean smaller fsaverage diagnostic separation. This comparison changes anatomy and test/reference sensor geometry; it does not isolate anatomy alone. Only no-warp primary peaks are used; no sensitivity variants are mixed in.')
        table(ds.drop(columns='cohort'))
        for image_path in sorted((out/cohort).glob('*.png')):
            rel=image_path.relative_to(out).as_posix()
            caption=image_path.stem.replace('_',' ')
            md.append(f'![{caption}]({rel})\n')
            ht.append(f'<figure><img loading="lazy" src="{rel}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    pd.concat(summaries).to_csv(tables/'03_method_summary.csv',index=False)
    pd.concat(sensitivities).to_csv(tables/'04_test_run_reference_sensitivity.csv',index=False)
    pd.concat(deltas).to_csv(tables/'05_equal_subject_paired_deltas.csv',index=False)
    coverage=paired.groupby(['strategy','_merge','jointly_pass'],observed=True).size().rename('count').reset_index()
    coverage.to_csv(tables/'06_pairing_coverage.csv',index=False)
    heading('Validation, reproducibility and next steps')
    paragraph('The report verified the evaluation source hashes, exact 5,124 unique job keys, all 12 methods, PASS/FAIL distance availability and uniqueness of the 732 historical baseline rows. Reconstruction inputs were verified by the evaluation stage. Plot input tables and matching coverage are supplied in tables/. Files are generated in a fresh directory and reconstruction/evaluation inputs are not modified.')
    paragraph('The no-warp plotting source supplies the original method order, year labels, family bands and box styles. PNG figures use 300 dpi. Master boxplots show outliers; comparison boxplots hide outlier markers using the same 1.5-IQR whiskers (values are retained in the tables). The all-seven view is an inclusive exploratory description, not a substitute for the separately reported QA cohorts.')
    paragraph('Next: review sensor-to-template alignment and independent ground-truth mapping before interpreting anatomical localization accuracy. The existing MRI registration outputs have not been applied by this pipeline. Re-running solvers is not required to inspect these diagnostic results.')
    for path,expected in hashes.items():
        require(sha(path)==expected,f'Input changed while plotting: {path}')
    (out/'REPORT.md').write_text('\n'.join(md).rstrip()+'\n')
    css='body{font:16px/1.55 system-ui,sans-serif;max-width:1450px;margin:40px auto;padding:0 24px;color:#202630}h1,h2{line-height:1.2}h2{margin-top:46px}table{border-collapse:collapse;font-size:13px;width:100%;margin:20px 0}td,th{padding:7px 9px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf1f6}tr:nth-child(even){background:#fafbfd}img{width:100%;height:auto}figure{margin:35px 0}figcaption{color:#566;font-size:13px}'
    (out/'REPORT.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Fsaverage diagnostic report</title><style>'+css+'</style></head><body>'+'\n'.join(ht)+'</body></html>')
    provenance=dict(created_utc=datetime.now(timezone.utc).isoformat(),input_sha256=hashes,
        rows=len(data),status_counts=data.status.value_counts().to_dict(),
        cohorts={k:v[1] for k,v in COHORTS.items()},dpi=300,metric='diagnostic unregistered HEAD separation mm',
        anatomical_error_valid=False,figures=sorted(str(x.relative_to(out)) for x in out.rglob('*.png')),
        no_warp_pairing_coverage=coverage.astype({'_merge':str}).to_dict('records'))
    (out/'07_plot_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (out/'README.txt').write_text('Open REPORT.html or REPORT.md. PNG-only figures follow the no-warp experiment style.\nThree cohort folders include all seven, other six, and exploratory sub-03.\nTables retain failures and provide equal-subject and matched no-warp summaries.\nAll fsaverage distances remain diagnostic; anatomical accuracy has not been validated.\n')
    print(f'Created {len(provenance["figures"])} PNGs, summary CSVs and REPORT.md/REPORT.html in {out}')


if __name__=='__main__':
    main()
