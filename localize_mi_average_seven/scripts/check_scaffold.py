"""Lightweight schema, schedule, PASS/FAIL, pairing and no-overwrite checks."""
import ast
from pathlib import Path
import tempfile

import pandas as pd
from common import config, EXP, ROOT, KEY, FINGERPRINTS, schedule, evaluate


def main():
    for path in (EXP / 'scripts').glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    c = config(EXP / 'config/experiment.json')
    original = schedule(c)
    assert len(original) == 5124
    assert len(original[['test_subject','run']].drop_duplicates()) == 61
    assert 'sub-03' in set(original.test_subject)
    with tempfile.TemporaryDirectory(prefix='average-seven-check-') as name:
        tmp = Path(name)
        # One paired success and one explicit sub-03 alignment failure.
        keys = [original.iloc[0][KEY].to_dict(), original[original.test_subject.eq('sub-03')].iloc[0][KEY].to_dict()]
        pd.DataFrame(keys).to_csv(tmp/'schedule.csv',index=False)
        # schedule() requires all seven subjects; retain one planned job each.
        small = original.groupby('test_subject',sort=False).head(1)[KEY]
        small.to_csv(tmp/'schedule.csv',index=False)
        c['schedule'] = str(tmp/'schedule.csv')
        records = []
        for arm in c['arms']:
            for key in small.to_dict('records'):
                records.append(dict(**key,arm=arm,status='PASS',failure_reason='',
                    alignment_status='FAIL' if key['test_subject']=='sub-03' else 'PASS',
                    coordinate_frame='model_head',coordinate_units='mm',
                    estimated_head_x_mm=3 if arm=='average_seven' else 0,estimated_head_y_mm=4,estimated_head_z_mm=0,
                    **{f:'same' for f in FINGERPRINTS}))
        # Drop a result to ensure a missing job remains a FAIL.
        records.pop()
        pd.DataFrame(records).to_csv(tmp/'results.csv',index=False)
        refs = small[['test_subject','run']].assign(coordinate_frame='test_head',coordinate_units='mm',
            gt_head_x_mm=0,gt_head_y_mm=0,gt_head_z_mm=0,provenance='synthetic original reference')
        refs.to_csv(tmp/'refs.csv',index=False)
        evaluate(c,tmp/'out',tmp/'results.csv',tmp/'refs.csv')
        rows = pd.read_csv(tmp/'out/01_all_outcomes.csv')
        pairs = pd.read_csv(tmp/'out/02_paired_differences.csv')
        assert len(rows)==14 and len(pairs)==7
        assert not rows.localization_error_valid.any()
        assert 'error_mm' not in rows.columns
        assert rows[rows.test_subject.eq('sub-03')].evaluation_failure_reason.eq('ALIGNMENT_FAIL').all()
        assert rows.evaluation_failure_reason.eq('MISSING_RESULT').sum()==1
        assert pairs[pairs.paired_eligible].diagnostic_difference_mm.eq(1).all()
        # Native scanner frames and metre-valued coordinates cannot masquerade as head mm.
        for field, value, reason in [('coordinate_frame','native_scanner_ras_mm','FRAME_MISMATCH'),
                                      ('coordinate_units','m','UNIT_MISMATCH')]:
            invalid = pd.DataFrame(records)
            invalid.loc[0, field] = value
            invalid.to_csv(tmp/'invalid.csv', index=False)
            evaluate(c,tmp/field,tmp/'invalid.csv',tmp/'refs.csv')
            rejected = pd.read_csv(tmp/field/'01_all_outcomes.csv')
            assert rejected.evaluation_failure_reason.eq(reason).sum() == 1
        mismatch = pd.DataFrame(records)
        mismatch.loc[0, 'configuration_key'] = 'different'
        mismatch.to_csv(tmp/'mismatch.csv',index=False)
        evaluate(c,tmp/'mismatch',tmp/'mismatch.csv',tmp/'refs.csv')
        unmatched = pd.read_csv(tmp/'mismatch/02_paired_differences.csv')
        assert not unmatched.iloc[0].paired_eligible
        assert pd.isna(unmatched.iloc[0].diagnostic_difference_mm)
        try:
            evaluate(c,tmp/'out',tmp/'results.csv',tmp/'refs.csv')
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing output was overwritten')
    print('PASS: syntax, 61-run/all-seven schedule, statuses, diagnostic head-mm distances, pairing, no overwrite')


if __name__=='__main__':
    main()
