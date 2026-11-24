"""Check candidate fiducial GUI inputs; --open launches manual placement."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open', action='store_true', help='Open the MNE GUI')
    args = parser.parse_args()
    exp = Path(__file__).resolve().parents[1]
    candidate = exp/'work/bem_candidates/local_inner_1mm'
    subjects = candidate/'validation_subjects'
    subject = subjects/'average_seven'
    validation = json.loads((candidate/'validation.json').read_text())
    if validation.get('bem_solution') != 'PASS':
        raise ValueError('Candidate must pass BEM validation')
    for name in ['bem/outer_skin.surf', 'mri/T1.mgz', 'mri/transforms/talairach.xfm']:
        if not (subject/name).is_file():
            raise FileNotFoundError(subject/name)
    import mne
    # This checks the GUI's scalp and initial-fiducial inputs without opening it.
    model = mne.coreg.Coregistration(info=None, subject='average_seven',
                                    subjects_dir=subjects, fiducials='auto')
    target = subject/'bem/average_seven-fiducials.fif'
    print('Candidate GUI inputs loaded.')
    print(f'Save manually reviewed MRI fiducials to: {target}')
    print('Any automatically estimated landmarks are starting guesses, not reviewed landmarks.')
    print('Unlock fiducials, place LPA/Nasion/RPA on the template, then use Save MRI Fid.')
    print('Keep anatomy scaling disabled. This session does not run EEG coregistration.')
    print('Anatomy approval remains pending; no experiment manifest is changed.')
    del model
    if args.open:
        mne.viz.set_3d_backend('pyvistaqt')
        mne.gui.coregistration(subject='average_seven', subjects_dir=str(subjects),
                               head_high_res=False, block=True)
    else:
        print('Input check only. Add --open to launch manual placement.')


if __name__ == '__main__':
    main()
