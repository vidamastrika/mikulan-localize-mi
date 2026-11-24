"""Prepare matched regional overlays and an anatomical approval form."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from review_common import candidate, DIAGNOSTICS, FSAVERAGE, ROOT, VAR, new_run, verified_diagnostics, write_json
from diagnose import COLORS, MRI, overlay_figure, surfaces_for

FOCUSES = {
    'vault': [0, 0, 35],
    'face_orbits': [0, 45, -15],
    'inferior_skull_base': [0, 0, -35],
    'left_lateral': [-55, 0, 25],
    'right_lateral': [55, 0, 25],
}


def flagged(row):
    return (row.get('status', '').startswith(('UNRESOLVED', 'WITHHELD')) or
            row.get('intersection_pairs', 0) > 0 or row.get('inside_fraction', 1) < 1 or
            not row.get('sign_topology_eligible', True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    with new_run(args.output_root, '08_ANATOMICAL_REVIEW') as run:
        subject = verified_diagnostics(run)
        t1, surfaces = candidate(subject)
        config = json.loads(run.use(VAR / 'config/variant.json').read_text())
        dataset = ROOT / config['dataset']
        diag = json.loads((DIAGNOSTICS / 'manifest.json').read_text())
        screens = json.loads((DIAGNOSTICS / '04_geometry_screens.json').read_text())
        flags = [r for r in screens if flagged(r)]
        pd.DataFrame(flags).drop(columns='first_pairs', errors='ignore').to_csv(
            run.out / '01_flagged_geometry.csv', index=False)
        explanations = []
        for row in flags:
            name = f"{row['model']} {row.get('layer') or row.get('inner', '')}"
            if row['check'] == 'nesting':
                outside = round(row['vertices_tested'] * (1 - row['inside_fraction']))
                explanations.append(f"- {name} inside {row['outer']}. {outside} of "
                                    f"{row['vertices_tested']:,} vertices classified outside.")
            else:
                target = f" versus {row['outer']}" if 'outer' in row else ' against itself'
                explanations.append(f"- {name}{target}. {row['intersection_pairs']} triangle pairs "
                                    f"flagged among {row['candidate_pairs_tested']:,} tested pairs.")
        models = [('maskface_baseline', t1, surfaces, False),
                  ('fsaverage', FSAVERAGE / 'mri/T1.mgz',
                   {layer: FSAVERAGE / 'bem' / f'{layer}.surf' for layer in surfaces if layer != 'brain'}, False)]
        for name in config['participants']:
            models.append((name, dataset / config['original_T1w_pattern'].format(subject=name),
                           {layer: dataset / 'derivatives/sourcemodelling' / name / 'anat'
                            / f'{name}_{layer}.surf.gii' for layer in surfaces if layer != 'brain'}, True))
        figures = run.out / 'figures'
        figures.mkdir()
        slices = []
        for name, path, paths, native in models:
            print(f'Plotting {name}', flush=True)
            # Check the exact MRI and meshes used in the diagnostic run.
            for item in (path, *paths.values()):
                run.use(item, diag['input_sha256'][str(item.resolve())])
            meshes = surfaces_for(run, paths, path, native)
            mri = MRI(path, np.eye(4))
            anchor = np.array(diag['region_anchors_scanner_ras_mm'][name])
            label = name
            outlines = {label: [(layer, *mesh, COLORS[layer]) for layer, mesh in meshes.items()
                                if layer != 'brain']}
            for region, offsets in FOCUSES.items():
                focus = anchor + offsets
                overlay_figure([(label, mri)], outlines, figures / f'{name}_{region}.png',
                               f'{name} | {region} | own MRI, matched offsets in scanner RAS', 200, focus)
                slices.append(dict(model=name, region=region, sagittal_x_mm=focus[0],
                                   coronal_y_mm=focus[1], axial_z_mm=focus[2]))
        pd.DataFrame(slices).to_csv(run.out / '02_slice_positions.csv', index=False)
        gaps = pd.read_csv(DIAGNOSTICS / '02_regional_candidate_comparison.csv')
        gaps[gaps.model.isin(n for n, *_ in models)].to_csv(run.out / '03_regional_comparison.csv', index=False)
        form = dict(candidate='maskface_baseline', decision='PENDING', reviewer='', review_date='',
                    anatomical_rationale='', review_manifest_sha256='',
                    regional_decisions=dict.fromkeys(['cranial_vault', 'face_orbits', 'inferior_skull_base',
                                                     'frame_and_layer_correspondence'], 'PENDING'),
                    input_sha256={str(p.resolve()):run.inputs[str(p.resolve())]
                                  for p in (t1, *surfaces.values())})
        write_json(run.out / 'APPROVAL_PENDING.json', form)
        counts = Counter(r['model'] for r in flags)
        (run.out / 'REPORT.md').write_text(
            '# Provisional maskface anatomy review\n\n'
            'Review is pending. Use the vault, face/orbits, skull base and bilateral lateral '
            'views for each model. The same offsets from each inner-skull median define the '
            'slice planes. This matches head-relative regions without claiming anatomical '
            'registration. The table records the absolute scanner-RAS locations. Compare '
            'the yellow outer skull with visible cranial boundaries and the cyan scalp, '
            'especially where the contours descend around the face and inferior head. '
            'Nearest-vertex gaps are geometry proxies.\n\n'
            f'## Geometry flags ({len(flags)})\n\n' + '\n'.join(explanations) + '\n\n'
            f'The maskface baseline has {counts["maskface_baseline"]} flagged rows. '
            'The reference flags are completed geometry screens. The Pydeface skull-pair '
            'triangle contacts and outside vertices are related findings. Inner-skull '
            'self-contact counts refer to triangle pairs and depend on the intersection '
            'test tolerance. They do not establish the extent of a defect or validate '
            'another model. Review the maskface boundary on its MRI independently.\n\n'
            '## Decision\n\n'
            'Record whether each region and the layer correspondence support exploratory '
            'BEM use. Explain deviations at the orbital and inferior boundaries. Keep '
            'the decision pending if tissue correspondence is uncertain.\n')
        run.doc.update(candidate='maskface_baseline', anatomical_approval='PENDING')


if __name__ == '__main__':
    main()
