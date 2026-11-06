# Draft request — not sent

Subject: Localize-MI: physical HydroCel sensor numbering in released EEG channels

We are evaluating a template-only reconstruction using Localize-MI EEG and
fsaverage, without using participants' MRI or digitized electrode positions.
Your paper identifies a 256-channel HydroCel CleanLeads net. The public
`import_egi_sdk` function assigns the saved EEG rows names e1 through e256.

Could you confirm:

1. For all seven subjects, does released channel **eN** correspond to physical
   HydroCel sensor **N** for every N=1…256, with no permutation introduced by the
   custom C++/MATLAB acquisition/export software?
2. Is the vertex/Cz acquisition reference excluded from these 256 EEG channels,
   and specifically is e256 physical sensor 256 rather than an inserted Cz row?
3. Was there any subject-specific net revision, channel reordering, omitted
   channel or special wiring that changes this correspondence? If available,
   could you point us to a channel-map export or acquisition code documenting it?

We can supply our pinned template filename and hash for reference. We are asking
about channel identity, not requesting measured positions, identifiable anatomy
or permission to infer alignment accuracy.
