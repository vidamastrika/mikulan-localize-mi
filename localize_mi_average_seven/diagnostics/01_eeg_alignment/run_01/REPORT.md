# EEG alignment diagnostic report

Status: released-data diagnostics complete; anatomical and coordinate-convention review pending.

## Verified numerical findings

Saved nominal numerical passes reproduced: {'average_seven': 3, 'fsaverage': 6}. Thresholds remain median ≤10 mm and p95 ≤20 mm, nearest vertex, all 256 sensors. Visual review remains required.

All-participant sidecar landmark identity: True. See pairwise electrode/landmark comparisons in 09_cross_participant_identity.csv.

Original forward sensor coordinates are recovered without refitting in 02_recovered_sensors.csv. HDF5 versus inverse-forward checks and GIFTI volume-center checks are in 01_coordinate_audit.csv. These test consistency of released artifacts, not independent digitizer provenance.

## Model and regional comparisons

subject         model            metric numerical_pass   n  median_mm    p95_mm   mean_mm    max_mm
 sub-01        native    nearest_vertex           True 256   9.879447 16.754867 10.189708 31.921623
 sub-01        native point_to_triangle     NOT_A_GATE 256   9.760848 16.726266 10.056742 31.887418
 sub-01     fsaverage    nearest_vertex           True 256   3.077724  9.425508  3.960126 19.994967
 sub-01     fsaverage point_to_triangle     NOT_A_GATE 256   2.595010  9.391346  3.495118 19.994967
 sub-01 average_seven    nearest_vertex          False 256   6.748605 23.617232  8.590529 44.825744
 sub-01 average_seven point_to_triangle     NOT_A_GATE 256   6.658349 23.501928  8.397738 44.824946
 sub-02        native    nearest_vertex          False 256  10.563524 22.680811 10.897451 30.689226
 sub-02        native point_to_triangle     NOT_A_GATE 256  10.389282 22.632524 10.730227 30.629439
 sub-02     fsaverage    nearest_vertex           True 256   5.078254 12.038023  5.547599 18.113002
 sub-02     fsaverage point_to_triangle     NOT_A_GATE 256   4.950667 12.006851  5.251291 17.958816
 sub-02 average_seven    nearest_vertex          False 256   9.227838 30.643302 11.631460 45.347540
 sub-02 average_seven point_to_triangle     NOT_A_GATE 256   9.186937 30.614633 11.455958 45.275872
 sub-03        native    nearest_vertex          False 256   9.763842 20.690681 10.668896 34.717662
 sub-03        native point_to_triangle     NOT_A_GATE 256   9.665179 20.587141 10.551875 34.717584
 sub-03     fsaverage    nearest_vertex          False 256   4.565580 22.597474  7.349045 44.678014
 sub-03     fsaverage point_to_triangle     NOT_A_GATE 256   4.304208 22.563567  6.997403 44.616826
 sub-03 average_seven    nearest_vertex          False 256   9.972396 39.392466 13.206486 60.304805
 sub-03 average_seven point_to_triangle     NOT_A_GATE 256   9.907556 39.381436 13.013662 60.304728
 sub-04        native    nearest_vertex          False 256  10.774696 16.529300 10.495212 20.957125
 sub-04        native point_to_triangle     NOT_A_GATE 256  10.737637 16.506321 10.370792 20.900318
 sub-04     fsaverage    nearest_vertex           True 256   3.631899 15.358994  5.260285 26.034211
 sub-04     fsaverage point_to_triangle     NOT_A_GATE 256   3.286730 15.261541  4.856509 25.989238
 sub-04 average_seven    nearest_vertex          False 256   6.901316 34.085662 10.780289 55.944545
 sub-04 average_seven point_to_triangle     NOT_A_GATE 256   6.872952 34.080651 10.575967 55.943655
 sub-05        native    nearest_vertex          False 256  10.092732 18.589829 10.145890 21.176773
 sub-05        native point_to_triangle     NOT_A_GATE 256  10.009507 18.558545  9.996596 21.176773
 sub-05     fsaverage    nearest_vertex           True 256   3.823550 11.412626  4.903889 17.147701
 sub-05     fsaverage point_to_triangle     NOT_A_GATE 256   3.653057 11.227934  4.505281 16.929123
 sub-05 average_seven    nearest_vertex           True 256   5.069618 15.179636  6.399145 41.203422
 sub-05 average_seven point_to_triangle     NOT_A_GATE 256   4.744562 15.135305  6.091829 41.197638
 sub-06        native    nearest_vertex          False 256  10.265341 16.809003 10.332250 29.628610
 sub-06        native point_to_triangle     NOT_A_GATE 256  10.157516 16.776179 10.225755 29.558883
 sub-06     fsaverage    nearest_vertex           True 256   2.653947  8.219870  3.327863 13.905640
 sub-06     fsaverage point_to_triangle     NOT_A_GATE 256   2.256017  7.983641  2.796480 13.793272
 sub-06 average_seven    nearest_vertex           True 256   4.972871 17.029766  6.252192 23.995984
 sub-06 average_seven point_to_triangle     NOT_A_GATE 256   4.761717 16.863222  6.000762 23.875953
 sub-07        native    nearest_vertex          False 256  10.249696 14.623192  9.889621 20.662209
 sub-07        native point_to_triangle     NOT_A_GATE 256  10.183806 14.601725  9.777582 20.654658
 sub-07     fsaverage    nearest_vertex           True 256   2.451862  8.021143  3.167208 13.623087
 sub-07     fsaverage point_to_triangle     NOT_A_GATE 256   2.077461  7.832021  2.651417 13.591378
 sub-07 average_seven    nearest_vertex           True 256   5.027063 14.512664  6.301466 25.814876
 sub-07 average_seven point_to_triangle     NOT_A_GATE 256   4.889420 14.438063  6.024859 25.777013

Regional concentration is reported continuously as fraction >20 mm and share of squared triangle residual, with region counts. A high share in a small region supports localized mismatch; broad elevations support global mismatch. No new pass/fail or concentration threshold is introduced.

subject         model          region  n  over_20mm  fraction_over_20mm  share_of_total_squared_triangle_residual  median_mm    p95_mm   mean_mm    max_mm
 sub-01 average_seven  anterior_proxy 27          7            0.259259                                  0.381018  13.561576 42.739572 18.087449 44.824946
 sub-01 average_seven      left_proxy 50         11            0.220000                                  0.368318  12.862914 27.873341 13.291095 37.809736
 sub-01 average_seven posterior_proxy 50          1            0.020000                                  0.090983   6.497522 12.774185  6.711726 21.960479
 sub-01 average_seven     right_proxy 45          1            0.022222                                  0.110919   7.751387 19.180208  7.818671 23.024667
 sub-01 average_seven  superior_proxy 84          0            0.000000                                  0.048762   3.367916  8.044671  3.684269 10.009066
 sub-01     fsaverage  anterior_proxy 27          0            0.000000                                  0.206119   5.017442 11.947355  5.379175 13.018509
 sub-01     fsaverage      left_proxy 50          0            0.000000                                  0.240403   3.399305  9.879528  4.042978 11.074215
 sub-01     fsaverage posterior_proxy 50          0            0.000000                                  0.208062   2.259784  8.130021  3.310505 19.994967
 sub-01     fsaverage     right_proxy 45          0            0.000000                                  0.049696   1.750947  4.097626  1.951416  8.043373
 sub-01     fsaverage  superior_proxy 84          0            0.000000                                  0.295719   2.876836  8.743323  3.500294 11.096551
 sub-01        native  anterior_proxy 27          0            0.000000                                  0.031711   4.811293  8.657944  5.334285 13.234755
 sub-01        native      left_proxy 50          0            0.000000                                  0.189268  10.176309 15.064464 10.285889 16.592627
 sub-01        native posterior_proxy 50          3            0.060000                                  0.333607  13.384307 19.726519 13.552476 31.887418
 sub-01        native     right_proxy 45          0            0.000000                                  0.111281   8.119907 12.138351  8.312509 13.737134
 sub-01        native  superior_proxy 84          0            0.000000                                  0.334133   9.954731 16.647823 10.291894 19.891792
 sub-02 average_seven  anterior_proxy 30          4            0.133333                                  0.182089  12.407765 38.860519 15.826030 45.275872
 sub-02 average_seven      left_proxy 46         25            0.543478                                  0.461673  20.439148 35.744579 21.755193 41.240958
 sub-02 average_seven posterior_proxy 45          5            0.111111                                  0.147500  11.251778 21.451698 11.823853 25.710544
 sub-02 average_seven     right_proxy 52          5            0.096154                                  0.166821   9.444512 27.296108 10.800120 36.626108
 sub-02 average_seven  superior_proxy 83          0            0.000000                                  0.041917   3.812720  9.095595  4.379828 12.410920
 sub-02     fsaverage  anterior_proxy 30          0            0.000000                                  0.156525   6.677021 11.755915  6.931381 13.311654
 sub-02     fsaverage      left_proxy 46          0            0.000000                                  0.365820   7.976227 13.799186  8.206394 15.404095
 sub-02     fsaverage posterior_proxy 45          0            0.000000                                  0.176457   5.906968 10.667285  5.552664 11.126535
 sub-02     fsaverage     right_proxy 52          0            0.000000                                  0.172928   4.206803  9.696466  4.719968 17.958816
 sub-02     fsaverage  superior_proxy 83          0            0.000000                                  0.128269   2.724627  7.476208  3.175743 10.166245
 sub-02        native  anterior_proxy 30          1            0.033333                                  0.026909   3.533188 11.458743  4.422804 19.864602
 sub-02        native      left_proxy 46          2            0.043478                                  0.253797  14.323868 19.857750 14.216043 25.069229
 sub-02        native posterior_proxy 45         15            0.333333                                  0.411365  17.615369 25.977858 18.508943 30.629439
 sub-02        native     right_proxy 52          2            0.038462                                  0.188058  10.775402 18.125201 11.306548 25.239413
 sub-02        native  superior_proxy 83          0            0.000000                                  0.119872   5.400977 13.360176  6.499677 16.610369
 sub-03 average_seven  anterior_proxy 25         11            0.440000                                  0.328654  13.647908 56.807033 26.936225 60.304728
 sub-03 average_seven      left_proxy 51         30            0.588235                                  0.455600  24.025402 44.696772 24.647097 58.539571
 sub-03 average_seven posterior_proxy 54          0            0.000000                                  0.023320   4.763976 12.005559  4.919167 12.689231
 sub-03 average_seven     right_proxy 51          6            0.117647                                  0.139488  11.955240 27.281346 13.582819 29.497330
 sub-03 average_seven  superior_proxy 75          1            0.013333                                  0.052937   5.063827 13.817923  5.903082 20.643163
 sub-03     fsaverage  anterior_proxy 25          7            0.280000                                  0.221318   6.289592 24.964789 11.321995 43.284765
 sub-03     fsaverage      left_proxy 51         12            0.235294                                  0.614602  15.613283 27.597745 16.203848 44.616826
 sub-03     fsaverage posterior_proxy 54          0            0.000000                                  0.032045   1.737811  8.106965  2.879321 12.661131
 sub-03     fsaverage     right_proxy 51          0            0.000000                                  0.057066   4.130621 10.185658  4.483724 13.339980
 sub-03     fsaverage  superior_proxy 75          0            0.000000                                  0.074970   3.325910 10.809908  3.969810 15.398384
 sub-03        native  anterior_proxy 25          1            0.040000                                  0.078498   7.905824 14.264027  9.253174 29.765945
 sub-03        native      left_proxy 51         12            0.235294                                  0.420828  14.246949 24.393741 15.764956 34.717584
 sub-03        native posterior_proxy 54          0            0.000000                                  0.161433   9.874734 14.191038  9.757456 17.694956
 sub-03        native     right_proxy 51          0            0.000000                                  0.098265   7.795986 12.152362  7.605183 16.038641
 sub-03        native  superior_proxy 75          1            0.013333                                  0.240976   9.731285 15.756703 10.015612 20.964567
 sub-04 average_seven  anterior_proxy 24         10            0.416667                                  0.382696  18.225667 53.062390 25.578972 55.943655
 sub-04 average_seven      left_proxy 46         14            0.304348                                  0.279579  13.272630 36.270171 16.032824 40.888313
 sub-04 average_seven posterior_proxy 44          0            0.000000                                  0.019726   4.429212  8.572147  4.578843 10.121995
 sub-04 average_seven     right_proxy 44         12            0.272727                                  0.266502  14.246399 33.958616 16.245246 41.209882
 sub-04 average_seven  superior_proxy 98          0            0.000000                                  0.051497   4.165689 10.055247  4.487576 17.731742
 sub-04     fsaverage  anterior_proxy 24          1            0.041667                                  0.318445  10.158338 18.810347 10.753486 25.989238
 sub-04     fsaverage      left_proxy 46          0            0.000000                                  0.236300   4.492917 14.120503  5.891503 15.268861
 sub-04     fsaverage posterior_proxy 44          0            0.000000                                  0.031650   1.260081  5.077943  2.043111 11.184605
 sub-04     fsaverage     right_proxy 44          0            0.000000                                  0.300756   6.167397 15.725014  7.311481 19.329548
 sub-04     fsaverage  superior_proxy 98          0            0.000000                                  0.112848   2.985753  6.662506  3.087464 11.121474
 sub-04        native  anterior_proxy 24          0            0.000000                                  0.081014  10.626107 15.700034  9.400180 18.138236
 sub-04        native      left_proxy 46          0            0.000000                                  0.128663   8.397369 16.203382  8.730366 17.507342
 sub-04        native posterior_proxy 44          0            0.000000                                  0.073338   6.698039 11.142855  6.860203 11.755394
 sub-04        native     right_proxy 44          0            0.000000                                  0.144466   9.380262 15.049241  9.491120 19.119856
 sub-04        native  superior_proxy 98          1            0.010204                                  0.572520  13.378711 16.817096 13.349626 20.900318
 sub-05 average_seven  anterior_proxy 18         12            0.666667                                  0.600745  25.823842 39.779677 22.508252 41.197638
 sub-05 average_seven      left_proxy 56          0            0.000000                                  0.101376   4.700166 11.421673  5.036201 14.218512
 sub-05 average_seven posterior_proxy 64          0            0.000000                                  0.146849   5.593349 13.289396  5.910903 13.944625
 sub-05 average_seven     right_proxy 44          0            0.000000                                  0.074651   4.557185 11.339796  4.865629 13.804158
 sub-05 average_seven  superior_proxy 74          0            0.000000                                  0.076379   3.945552  7.437785  3.783067 14.145979
 sub-05     fsaverage  anterior_proxy 18          0            0.000000                                  0.066815   4.338211  9.565395  4.547038 12.504849
 sub-05     fsaverage      left_proxy 56          0            0.000000                                  0.139365   3.339376  8.464727  3.683607 11.209648
 sub-05     fsaverage posterior_proxy 64          0            0.000000                                  0.369940   3.073696 15.236064  5.271489 16.929123
 sub-05     fsaverage     right_proxy 44          0            0.000000                                  0.102750   3.536017  7.501763  3.707721  8.506595
 sub-05     fsaverage  superior_proxy 74          0            0.000000                                  0.321130   4.338017 11.112882  4.928489 12.876535
 sub-05        native  anterior_proxy 18          0            0.000000                                  0.031976   6.650680 11.304260  6.818025 12.396172
 sub-05        native      left_proxy 56          2            0.035714                                  0.369672  14.454426 18.924065 14.402316 20.993559
 sub-05        native posterior_proxy 64          2            0.031250                                  0.386610  13.974961 19.058939 13.318946 21.176773
 sub-05        native     right_proxy 44          0            0.000000                                  0.027414   3.599488  8.320023  3.927330 10.024504
 sub-05        native  superior_proxy 74          0            0.000000                                  0.184328   8.735988 14.229219  8.171074 16.541367
 sub-06 average_seven  anterior_proxy 33          2            0.060606                                  0.211520   6.882185 20.465860  8.437890 23.875953
 sub-06 average_seven      left_proxy 52          1            0.019231                                  0.233474   6.550534 14.680848  7.159983 23.619842
 sub-06 average_seven posterior_proxy 48          0            0.000000                                  0.150269   4.515102 15.160719  5.676485 19.712256
 sub-06 average_seven     right_proxy 45          3            0.066667                                  0.372003   8.776812 20.085284  9.921899 21.326100
 sub-06 average_seven  superior_proxy 78          0            0.000000                                  0.032734   1.985163  5.028495  2.134215  5.566567
 sub-06     fsaverage  anterior_proxy 33          0            0.000000                                  0.036081   1.266622  4.040685  1.490425  5.655159
 sub-06     fsaverage      left_proxy 52          0            0.000000                                  0.202925   2.573141  6.596690  2.888298 13.793272
 sub-06     fsaverage posterior_proxy 48          0            0.000000                                  0.288282   2.483797  8.726310  3.450205 10.483124
 sub-06     fsaverage     right_proxy 45          0            0.000000                                  0.377582   4.563527  8.900174  4.674128  9.695575
 sub-06     fsaverage  superior_proxy 78          0            0.000000                                  0.095130   1.699150  3.392677  1.802278  4.735500
 sub-06        native  anterior_proxy 33          0            0.000000                                  0.175807  12.653993 15.836552 12.628366 18.163426
 sub-06        native      left_proxy 52          3            0.057692                                  0.233543  10.157516 19.212278 10.996136 29.558883
 sub-06        native posterior_proxy 48          4            0.083333                                  0.149857   6.220198 21.066649  8.303608 23.383880
 sub-06        native     right_proxy 45          0            0.000000                                  0.164498   8.807204 17.931601  9.658084 18.520783
 sub-06        native  superior_proxy 78          0            0.000000                                  0.276295  10.425264 13.553315 10.206041 14.225863
 sub-07 average_seven  anterior_proxy 36          1            0.027778                                  0.150294   6.356937 13.782958  6.934582 20.737340
 sub-07 average_seven      left_proxy 54          0            0.000000                                  0.102021   4.175370  8.959931  4.613007 10.125843
 sub-07 average_seven posterior_proxy 40          0            0.000000                                  0.133468   4.920265 13.006203  5.800587 15.735277
 sub-07 average_seven     right_proxy 47          1            0.021277                                  0.548744  12.351457 17.746248 12.788942 25.777013
 sub-07 average_seven  superior_proxy 79          0            0.000000                                  0.065473   1.872232  6.178636  2.664718 10.854640
 sub-07     fsaverage  anterior_proxy 36          0            0.000000                                  0.082539   1.523486  5.615924  1.957432  8.255809
 sub-07     fsaverage      left_proxy 54          0            0.000000                                  0.359984   2.495201 10.192894  3.511426 12.984141
 sub-07     fsaverage posterior_proxy 40          0            0.000000                                  0.107771   1.868038  5.825385  2.466004  6.299880
 sub-07     fsaverage     right_proxy 47          0            0.000000                                  0.293291   2.665851  8.815427  3.479059 13.591378
 sub-07     fsaverage  superior_proxy 79          0            0.000000                                  0.156414   1.819474  4.934121  1.981295  7.562164
 sub-07        native  anterior_proxy 36          0            0.000000                                  0.222046  13.034474 15.496552 12.880298 19.634114
 sub-07        native      left_proxy 54          0            0.000000                                  0.219295  10.923184 13.211170 10.264119 17.584787
 sub-07        native posterior_proxy 40          0            0.000000                                  0.080541   6.687470 12.066554  6.563472 15.730118
 sub-07        native     right_proxy 47          1            0.021277                                  0.169229   9.264372 13.919149  9.246691 20.654658
 sub-07        native  superior_proxy 79          0            0.000000                                  0.308889  10.193504 13.860244  9.974359 16.267262

## Coordinate provenance and unresolved conventions

FIF HEAD tags and metre units are verified file contracts. Their anatomical meaning is unresolved. GIFTI positions are treated as FreeSurfer surface RAS millimetres under the existing pipeline contract; magnitude checks do not independently prove units or orientation.

The exporter fx_bids.make_coordsys_eeg_json copies epoch dig fiducials and labels them T1w; make_electrodes_eeg_tsv copies epoch channel loc. This explains what the code writes, but does not prove which input files produced the release or why all participants repeat the same values. Shared montage/export input is a hypothesis, not a verified cause.

fx_preprocess.make_dig_files negates the first two Polhemus coordinates and writes HPTS/FCsv. Original digitizer records and the exact MNE montage import history are needed to verify units, handedness, and anatomical correspondence independently.

Released channel TSVs independently check label sets/order only. NPY epoch arrays contain signals, not independent sensor coordinates. Matching HDF5 and forward transforms can share provenance. No independent anatomical validation is claimed.

Sidecar landmarks are evaluated under two explicit hypotheses (surface RAS versus forward HEAD). Neither interpretation is selected. Template landmarks are estimated and unreviewed. No verified native anatomical fiducials are available; direct anatomical landmark correspondence remains unresolved.

Full extents, robust spans and horizontal sections compare all seven scalps with both templates. Sections use original surface RAS axes, without new MRI registration; orientation, surface coverage, facial masking and neck extent confound size/shape attribution.

Regions are native surface-RAS geometric proxies (superior, left, right, anterior, posterior). Each channel retains the same native bin across models. These are not atlas labels or independently validated anatomical regions.

Nearest-vertex distances reproduce the original gate. Exact unsigned triangle distances supplement them and do not redefine acceptance. Neither metric distinguishes inside from outside or measures digitization error. Plots use shared mm axes and 0–40 mm color scales; rendering alone is subsampled.

## Scope and preservation

No anatomy, saved transform, existing result, alignment threshold or cohort membership was changed. No scaling, ICP, forward calculation or inverse localization was run. The BEM repair remains pending anatomical review and is not validated by scalp proximity.

Inventory scope is the accessible dataset tree, selected templates and saved alignment provenance; absence here is not proof of absence elsewhere. See manifest.json for content hashes, software versions and output hashes.
