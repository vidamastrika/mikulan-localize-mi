# Interpretation of maskface provenance

The released sidecars cite Milchenko and Marcus, *Obscuring Surface Anatomy in Volumetric Imaging Data* (2013), DOI 10.1007/s12021-012-9160-3. [Primary full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC3538950/), inspected 2026-11-18.

The method family modifies a boundary layer through fill coating, local blur or normalized filtering. The latter smooths within a flattened surface layer and maps it back. The paper evaluates skull stripping and gain-field correction; those tests do not establish FreeSurfer or BEM validity here. It also describes potential brain intrusion in some cases and later protective masking. See Methods, Masking of the Boundary Layer, and Evaluation and Results.

The released sidecar does not identify the exact algorithm variant, software version, filtering parameters, altered-voxel mask or brain-protection settings. Additional signal should therefore be described as masked/modified image support, not verified original facial tissue. This experiment does not reconstruct any anatomy.

Raw image agreement in overlapping regions is direct evidence about those shared voxels. Pydeface-zero/maskface-positive voxels lack an unmodified paired reference and cannot be assigned original-versus-filtered truth from these two images alone.
