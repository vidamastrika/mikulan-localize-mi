# Scalp geometry and MRI investigation

Status: diagnostic measurements complete; anatomical assessment pending. No previous output or anatomy was modified.

## Confirmed file and numerical checks

Prior run COMPLETE with verified input/output hashes. Primary exact triangle distances reproduced for all 21 subject/model combinations. 31 scalar MRI records and 22 transform records audited.

Thresholds remain nearest-vertex median ≤10 mm and p95 ≤20 mm, with visual review. No fit, scaling, forward, inverse or new MRI registration was performed.

Construction shell and exact log excerpts are retained. FreeSurfer recon-all resampling/conformation evidence is in 16_FreeSurfer_processing_evidence.csv. Input hashes are verified against the completed construction manifest. Scalar MRI shapes, voxel sizes, qform/sform, FOV, intensity-support bounds and boundary support are exported. Log lines identify interpolation and repeated template shape updates where recorded.

## Inside/outside relative to estimated scalp

subject         model      classification   n
 sub-01 average_seven  INSIDE_CONDITIONAL  35
 sub-01 average_seven OUTSIDE_CONDITIONAL 218
 sub-01 average_seven          UNRESOLVED   3
 sub-01     fsaverage  INSIDE_CONDITIONAL 116
 sub-01     fsaverage OUTSIDE_CONDITIONAL 134
 sub-01     fsaverage          UNRESOLVED   6
 sub-01        native OUTSIDE_CONDITIONAL 255
 sub-01        native          UNRESOLVED   1
 sub-02 average_seven  INSIDE_CONDITIONAL  38
 sub-02 average_seven OUTSIDE_CONDITIONAL 218
 sub-02     fsaverage  INSIDE_CONDITIONAL  42
 sub-02     fsaverage OUTSIDE_CONDITIONAL 211
 sub-02     fsaverage          UNRESOLVED   3
 sub-02        native  INSIDE_CONDITIONAL   1
 sub-02        native OUTSIDE_CONDITIONAL 253
 sub-02        native          UNRESOLVED   2
 sub-03 average_seven  INSIDE_CONDITIONAL  26
 sub-03 average_seven OUTSIDE_CONDITIONAL 228
 sub-03 average_seven          UNRESOLVED   2
 sub-03     fsaverage  INSIDE_CONDITIONAL  44
 sub-03     fsaverage OUTSIDE_CONDITIONAL 208
 sub-03     fsaverage          UNRESOLVED   4
 sub-03        native  INSIDE_CONDITIONAL   1
 sub-03        native OUTSIDE_CONDITIONAL 255
 sub-04 average_seven  INSIDE_CONDITIONAL  38
 sub-04 average_seven OUTSIDE_CONDITIONAL 218
 sub-04     fsaverage  INSIDE_CONDITIONAL  70
 sub-04     fsaverage OUTSIDE_CONDITIONAL 180
 sub-04     fsaverage          UNRESOLVED   6
 sub-04        native OUTSIDE_CONDITIONAL 256
 sub-05 average_seven  INSIDE_CONDITIONAL  63
 sub-05 average_seven OUTSIDE_CONDITIONAL 190
 sub-05 average_seven          UNRESOLVED   3
 sub-05     fsaverage  INSIDE_CONDITIONAL 174
 sub-05     fsaverage OUTSIDE_CONDITIONAL  75
 sub-05     fsaverage          UNRESOLVED   7
 sub-05        native  INSIDE_CONDITIONAL   5
 sub-05        native OUTSIDE_CONDITIONAL 249
 sub-05        native          UNRESOLVED   2
 sub-06 average_seven  INSIDE_CONDITIONAL  21
 sub-06 average_seven OUTSIDE_CONDITIONAL 233
 sub-06 average_seven          UNRESOLVED   2
 sub-06     fsaverage  INSIDE_CONDITIONAL 164
 sub-06     fsaverage OUTSIDE_CONDITIONAL  88
 sub-06     fsaverage          UNRESOLVED   4
 sub-06        native OUTSIDE_CONDITIONAL 256
 sub-07 average_seven  INSIDE_CONDITIONAL  22
 sub-07 average_seven OUTSIDE_CONDITIONAL 232
 sub-07 average_seven          UNRESOLVED   2
 sub-07     fsaverage  INSIDE_CONDITIONAL 137
 sub-07     fsaverage OUTSIDE_CONDITIONAL 108
 sub-07     fsaverage          UNRESOLVED  11
 sub-07        native OUTSIDE_CONDITIONAL 256

Positive signed distance means outside; negative means inside. Labels are CONDITIONAL: closed single-component consistently oriented nondegenerate topology, absolute winding near 0 or 1, agreement of three oblique ray parities, and distance >0.1 mm. Other points remain UNRESOLVED. Full self-intersection absence is not certified. These signs describe the mesh, not anatomy or digitization correctness.

## Surface geometry

                       model  vertices  faces  vertices_with_fewer_than_three_faces  boundary_edges  nonmanifold_edges  inconsistent_oriented_edges  degenerate_faces  connected_components  unused_vertices  euler      area_mm2  signed_volume_mm3  sign_topology_eligible  self_intersection_certified   x_min_mm    y_min_mm    z_min_mm  x_max_mm   y_max_mm   z_max_mm   width_mm  ap_length_mm  height_mm                                                        coverage_convention
               average_seven     10242  20480                                     0               0                  0                            0                 0                     1                0      2 123672.033772       3.654506e+06                    True                        False -76.384270 -113.615311  -92.321426 78.015930  86.943123 120.788460 154.400200    200.558434 213.109886 area sectors about own median surface vertex; no intersubject registration
                   fsaverage     10242  20480                                     0               0                  0                            0                 0                     1                0      2 141198.095911       4.666789e+06                    True                        False -84.150574 -122.682236 -126.672989 85.864319  90.227631 104.891769 170.014893    212.909866 231.564758 area sectors about own median surface vertex; no intersubject registration
                      sub-01     10242  20480                                     0               0                  0                            0                 0                     1                0      2 141582.688787       4.550324e+06                    True                        False -79.892410 -115.239044 -127.716644 77.131409 101.698273 114.672470 157.023819    216.937317 242.389114 area sectors about own median surface vertex; no intersubject registration
                      sub-02     10242  20480                                     0               0                  0                            0                 0                     1                0      2 129825.983212       4.139612e+06                    True                        False -81.725548 -112.787430 -111.803436 80.402847  96.597717 107.988129 162.128395    209.385147 219.791565 area sectors about own median surface vertex; no intersubject registration
                      sub-03     10242  20480                                     0               0                  0                            0                 0                     1                0      2 146256.081525       4.830396e+06                    True                        False -81.876724 -110.773232 -125.772507 83.664886 106.725647 113.379852 165.541611    217.498878 239.152359 area sectors about own median surface vertex; no intersubject registration
                      sub-04     10242  20480                                     0               0                  0                            0                 0                     1                0      2 148057.853332       4.916397e+06                    True                        False -79.536682 -111.488380 -127.720764 82.012009 110.364563 109.159714 161.548691    221.852943 236.880478 area sectors about own median surface vertex; no intersubject registration
                      sub-05     10242  20480                                     0               0                  0                            0                 0                     1                0      2 125199.027410       3.746579e+06                    True                        False -78.064026 -107.537170 -126.050308 71.644501  93.409737 110.145554 149.708527    200.946907 236.195862 area sectors about own median surface vertex; no intersubject registration
                      sub-06     10242  20480                                     0               0                  0                            0                 0                     1                0      2 123604.071789       3.755834e+06                    True                        False -78.627441 -100.811142 -127.443413 74.100067  92.769135 107.685669 152.727509    193.580276 235.129082 area sectors about own median surface vertex; no intersubject registration
                      sub-07     10242  20480                                     0               0                  0                            0                 0                     1                0      2 123857.454632       3.767850e+06                    True                        False -78.751205 -101.698349 -127.738358 76.832855  92.016510 107.610764 155.584061    193.714859 235.349121 area sectors about own median surface vertex; no intersubject registration
average_seven_alternative_01     10242  20480                                     0               0                  0                            0                 0                     1                0      2 123672.033772       3.654506e+06                    True                        False -76.384270 -113.615311  -92.321426 78.015930  86.943123 120.788460 154.400200    200.558434 213.109886 area sectors about own median surface vertex; no intersubject registration
average_seven_alternative_02     10242  20480                                     0               0                  0                            0                 0                     1                0      2 123672.033772       3.654506e+06                    True                        False -76.384270 -113.615311  -92.321426 78.015930  86.943123 120.788460 154.400200    200.558434 213.109886 area sectors about own median surface vertex; no intersubject registration
average_seven_alternative_03     10242  20480                                     0               0                  0                            0                 0                     1                0      2 123672.033772       3.654506e+06                    True                        False -76.384270 -113.615311  -92.321426 78.015930  86.943123 120.788460 154.400200    200.558434 213.109886 area sectors about own median surface vertex; no intersubject registration
average_seven_alternative_04     10242  20480                                     0               0                  0                            0                 0                     1                0      2 123672.033772       3.654506e+06                    True                        False -76.384270 -113.615311  -92.321426 78.015930  86.943123 120.788460 154.400200    200.558434 213.109886 area sectors about own median surface vertex; no intersubject registration
average_seven_alternative_05     10242  20480                                     0               0                  0                            0                 0                     1                0      2 123672.033738       3.654506e+06                    True                        False -76.384269 -113.615312  -92.321426 78.015931  86.943120 120.788462 154.400200    200.558431 213.109888 area sectors about own median surface vertex; no intersubject registration
    fsaverage_alternative_06    232777 467036                                     0               0                714                            0                 0                     1                0    -27 150822.171963       4.836290e+06                   False                        False -83.963469 -122.499801 -125.499994 85.133068  89.720361 104.780540 169.096537    212.220162 230.280533 area sectors about own median surface vertex; no intersubject registration
    fsaverage_alternative_07     14934  29998                                     7              39                 97                            7                 0                     1                0    -14 150697.836500       4.835834e+06                   False                        False -83.941557 -122.488678 -125.500605 85.139088  89.679673 104.772501 169.080645    212.168351 230.273105 area sectors about own median surface vertex; no intersubject registration
    fsaverage_alternative_08     10242  20480                                     0               0                  0                            0                 0                     1                0      2 141198.095850       4.666789e+06                    True                        False -84.150575 -122.682236 -126.672983 85.864320  90.227634 104.891770 170.014895    212.909870 231.564753 area sectors about own median surface vertex; no intersubject registration

Dimensions are in unregistered surface RAS mm. Regional areas use angular sectors about each mesh median and fixed Z bands; medians, surface coverage, orientation, defacing and neck extent confound comparisons. Section extrema are exact triangle-plane intersections. No scaling or surface alignment is used.

## MRI boundary evidence and interpretation

15_regional_planarity_cues.csv adds area-weighted normal coherence, normal axis alignment and patch extents to identify potentially flattened patches for review. High coherence can also occur in normal anatomy.

        model    region   n  median_edge_offset_mm  fraction_profiles_in_fov
average_seven  anterior 245                  -2.75                  0.967347
average_seven      left 261                  -2.75                  0.957854
average_seven posterior 263                  -3.75                  0.954373
average_seven     right 255                  -2.75                  0.945098
    fsaverage  anterior 251                  -3.75                  0.976096
    fsaverage      left 259                  -4.25                  0.972973
    fsaverage posterior 265                  -4.25                  0.796226
    fsaverage     right 249                  -4.25                  0.951807
       sub-01  anterior 247                  -6.25                  0.955466
       sub-01      left 261                  -4.25                  0.957854
       sub-01 posterior 263                  -4.25                  0.923954
       sub-01     right 253                  -4.25                  0.944664
       sub-02  anterior 248                  -4.25                  1.000000
       sub-02      left 260                  -1.25                  1.000000
       sub-02 posterior 263                  -3.25                  1.000000
       sub-02     right 253                  -1.75                  1.000000
       sub-03  anterior 245                  -6.75                  0.963265
       sub-03      left 264                  -3.25                  0.984848
       sub-03 posterior 264                  -3.75                  1.000000
       sub-03     right 251                  -3.75                  0.972112
       sub-04  anterior 246                  -7.25                  0.934959
       sub-04      left 262                  -4.75                  0.961832
       sub-04 posterior 263                  -4.75                  0.958175
       sub-04     right 253                  -4.25                  0.944664
       sub-05  anterior 248                  -4.75                  0.975806
       sub-05      left 263                  -3.25                  0.980989
       sub-05 posterior 261                  -3.75                  0.973180
       sub-05     right 252                  -3.25                  0.980159
       sub-06  anterior 247                  -4.75                  0.967611
       sub-06      left 260                  -3.25                  0.984615
       sub-06 posterior 263                  -3.25                  0.969582
       sub-06     right 254                  -3.25                  0.976378
       sub-07  anterior 248                  -4.50                  0.975806
       sub-07      left 260                  -3.25                  0.984615
       sub-07 posterior 265                  -3.75                  0.966038
       sub-07     right 251                  -3.25                  0.972112

Radial edge offsets are strongest outward intensity drops within ±15 mm at 1024 deterministic surface vertices. They are review cues, not an independently validated scalp segmentation. Competing skull, tissue, hair, masking and bias-field boundaries can dominate. FOV support at 1%, 5%, 10% of positive-intensity p99 is also descriptive. A closed surface can still have an anatomically flattened or missing region.

Inspect average_seven_original_template_scalp.png, average_seven_FS_import_scalp.png and average_seven_MRI_scalp_overview.png together. A boundary already absent in original inputs/registered images suggests source coverage or construction effects; a visible MRI boundary outside the scalp across stages suggests segmentation mismatch. Expert slice review is required to distinguish these. Electrode distance alone cannot establish an anatomically invalid template.

Warp-field Jacobians are coarse field-only samples, not full-resolution or composite deformation validation. Retained warped images may precede the final template shape update; their normalized mean need not reproduce the final template. An absence of crop keywords does not prove cropping never occurred upstream.

## Coordinate conventions and figure reading

All panels use physical surface RAS millimetres: X right, Y anterior, Z superior. Native bridge reconstructed from GIFTI VolGeom scanner affine and tkregister geometry must match the actual released T1w affine/shape. Template bridges use each actual T1.mgz scanner affine and vox2ras_tkr. No unconditional cras-only approximation is made. See coordinate_bridges.json. These are header consistency checks, not independent electrode provenance.

ANTs affine/warp vector conventions are LPS; NIfTI affines are exposed by nibabel in RAS. Warp displacement statistics explicitly convert LPS vectors to RAS before physical Jacobian sampling. No ANTs transform is applied to electrodes.

Original template and imported MRI are sampled via scanner RAS using the FreeSurfer scalp bridge, permitting construction-versus-segmentation review. Trilinear interpolation is for diagnostic sampling/rendering only.

MRI residual figures center on the largest e238–e252 residual and the largest residual outside that block, for every participant and model. The plane passes through the electrode; the closest point and residual arrow are projections and nearest-point off-plane offsets are printed. Full 3D residual coordinates and MRI intensity paths are exported. Do not infer that both vector endpoints lie on the displayed slice.

Native fiducials retain two unverified sidecar interpretations from run_01; template fiducials are estimated/unreviewed. Missing retained native dig fiducials and repeated release sidecars remain unresolved. Native surfaces provide a reference, not independent sensor validation.

Alternative representations are compared without replacement. Representation agreement is not independent anatomical validation. Alternative exact sensor distances/signs are computed for e238–e252 plus each configuration’s five largest primary residuals, rather than treated as new alignment fits.

## Causal assessment

Confirmed: file/grid/topology/containment measurements, prior-distance reproduction and saved-transform reuse. Hypotheses needing review: template construction bias, source defacing/cropping, scalp segmentation collapse, and participant electrode convention errors. This script does not automatically select a cause from intensity edges or sensor distances.

Review the original/native MRI panels and construction stages before attributing failure. Any final causal finding should cite specific slice coordinates and evidence from both MRI and geometry. No BEM or alignment approval is granted.

Warnings: ["/home/girvi949/mne_data/MNE-fsaverage-data/fsaverage/bem/fsaverage-head-medium.fif: Surface outer skin  has topological defects: 42 / 14934 vertices have fewer than three neighboring triangles [3576, 3986, 6284, 7050, 7339, 10176, 11703]"]
