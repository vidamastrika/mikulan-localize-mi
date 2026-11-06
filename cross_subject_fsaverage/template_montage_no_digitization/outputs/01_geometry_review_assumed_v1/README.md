# Template-only scalp placement review

Status: PLACEMENT_CONCERN_REVIEW_REQUIRED. Classification counts: {'INSIDE': 191, 'OUTSIDE': 65, 'ON_SURFACE': 0, 'AMBIGUOUS': 0}.

01_scalp_overlay_views.png shows four views with the fixed electrodes and transparent scalp.
02_signed_scalp_distance.png shows signed distances and largest separations.
03_sensor_triangle_distance.csv preserves original coordinates and adds triangle distances/winding classes.
04_geometry_review.json records topology, numerical tolerances, hashes and interpretation limits.

Negative signed distance means inside, positive outside; ambiguous classifications retain missing signed distances.
Scalp topology was inspected, but self-intersections were not tested.
Original preparation thresholds and readiness files are unchanged. No sensors were relocated, projected or optimized.
No participant anatomy/digitization or ground-truth locations were read. No forward model was run or authorized.
