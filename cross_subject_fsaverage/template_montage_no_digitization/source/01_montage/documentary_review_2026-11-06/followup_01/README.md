# Follow-up: acquisition format and repository history

Outcome: **BLOCKED remains appropriate under the original evidence rules.**
The original preparation reports and prior documentary review were not changed.
No montage was selected by fitting or by localization results. Script 01 was not
re-executed, and no geometry QA or Script 02 was run.

## Additional checks completed

1. Inspected all 16 public default-branch revisions returned by the authors'
   repository commit API, including the initial 2019 publication and latest 2022
   revision. Each recursive file tree was retrieved completely. There are 20
   distinct paths across these revisions, including a renamed demo notebook;
   no apparent custom C++/MATLAB acquisition exporter or physical channel lookup
   is present. This conclusion is restricted to that public branch history.
   `repository_history_audit.json` records commit/tree hashes and inspected paths.
   Repository: https://github.com/iTCf/mikulan_et_al_2020

2. Inspected the related public `iTCf/ccepcoreg` main-branch tree, revision
   `8721fc04328c00b131659915dfbfa569bd342b54`. It contains a README, demonstration
   notebook, `export_bids.py` and `fx_bids.py`; no apparent acquisition-layer map
   was identified. This is a related dataset, not evidence that Localize-MI used
   exactly the same acquisition pipeline.
   https://github.com/iTCf/ccepcoreg

3. Obtained the manufacturer's Amp Server Pro SDK 3.0 Programmer's Reference
   Manual, document 8200031-01, May 2024. The readable original PDF is archived
   here. Pages 15–19 describe packet-format-dependent streaming: 256 EEG sample
   entries, separate reference-monitor data, and distinct GSN/HCGSN net codes.
   This supports keeping the reference monitor distinct from the EEG array and
   distinguishing net families. It does not establish which firmware/format or
   export permutation was used in the earlier Localize-MI acquisition.
   https://www.egi.com/images/stories/manuals/amp-server-pro-sdk-3-0-network-apis-user-guide-rev-01.pdf

4. Inspected the released Localize-MI top-level README and dataset description.
   They point to the same public source repository, not an additional acquisition
   map. The README lists Ezequiel Mikulan and Andrea Pigorini as dataset contacts.
   These are the contacts recorded in the downloaded dataset; current mailbox
   availability has not been tested. No message or GitHub issue was sent.

## Exact remaining question

For every participant, does released EEG **eN** correspond to physical
**HydroCel sensor N** for N=1…256, without an acquisition/export permutation or
an inserted vertex/reference channel? A short author confirmation or an original
channel-map export would resolve this specific missing link. A generic EGI
streaming convention alone cannot establish what a custom exporter saved.

The existing `../AUTHOR_CONFIRMATION_REQUEST.md` is ready for the user to send.
That question does not request individual anatomy, digitization or localization
outcomes. Do not fabricate an approval JSON or label the correspondence verified
before evidence answers it.

## Work that remains useful before Script 02

Once channel identity is established, pin the documented HydroCel template and
complete the evidence record. The fixed normalization and template-fiducial
placement are declared model assumptions, distinct from verified acquisition
identity. Then the user can run Script 01 into new source/output directories to
inspect numerical geometry. No automatic rerun or gate bypass is warranted now.
