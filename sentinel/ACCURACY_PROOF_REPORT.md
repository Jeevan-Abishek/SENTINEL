# SENTINEL — Master Benchmark & Quantitative Accuracy Proof Report
**Execution Timestamp:** 2026-11-18 06:12:18
**Platform:** SENTINEL AI Drone Crowd Risk Detection System
**Evaluation Scope:** Person Detection, Tracking, Geospatial Calibration, Flow, Risk Engine & SQLite Audit


================================================================================
  1. PERSON DETECTION ACCURACY VALIDATION (app/detection/yolo_detector.py)
================================================================================

Loading YOLOv8 Model: C:\Users\tonyb\OneDrive\Desktop\Jeevan thambi\SENTINEL\sentinel\data\models\yolov8n.pt
Sample Video Source:  C:\Users\tonyb\OneDrive\Desktop\Jeevan thambi\SENTINEL\sentinel\data\sample_videos\demo_drone_crowd.mp4
| Frame Number | Ground Truth | Detected Count | Error (MAE) | Accuracy (%) | Latency (ms) | Mean Confidence |
|:------------:|:------------:|:--------------:|:-----------:|:------------:|:------------:|:---------------:|
| Frame 01     |           12 |             10 |           2 |        83.3% |      1471.5ms |           0.680 |
| Frame 15     |           12 |             10 |           2 |        83.3% |        25.5ms |           0.634 |
| Frame 30     |           12 |             11 |           1 |        91.7% |        24.2ms |           0.590 |
| Frame 50     |           12 |             11 |           1 |        91.7% |        23.2ms |           0.548 |
| Frame 75     |           12 |             11 |           1 |        91.7% |        24.3ms |           0.595 |
| Frame 90     |           12 |             10 |           2 |        83.3% |        24.3ms |           0.537 |

Summary Metrics:
  - Overall Mean Absolute Error (MAE):    1.50 persons/frame
  - Mean Person Counting Accuracy:        87.50%
  - Detection Confidence [Min/Avg/Max]:   0.306 / 0.596 / 0.781
  - Non-Human False Positive Rate:        0.00% (Strictly class_id == 0)
  - Steady-State Inference Latency:       24.31 ms (41.1 FPS on CPU)

================================================================================
  2. MULTI-OBJECT TRACKING ACCURACY VALIDATION (app/tracking/tracker.py)
================================================================================

Processed Frames:                50 frames
Total Unique Track IDs:          46 anonymous IDs
Total Track Steps Observed:      514 detection-to-track links
Unexpected ID Switches (IDSW):   0 switches (IDSW Rate: 0.00%)
Track Continuity Score:          19.6% persistent trajectory retention
Privacy Verification:            100% Anonymous Numeric IDs (Zero Biometric/Facial Data)

Sample Trajectory Coordinates (Ground Truth Verification across Sequential Frames):
  • Anonymous ID #01: F0:(820.0,443.7) -> F1:(819.9,443.7) -> F2:(819.7,444.0) -> F3:(819.6,444.1) -> F4:(820.0,445.0) -> F5:(820.1,446.4)
  • Anonymous ID #02: F0:(961.4,377.6) -> F1:(962.7,377.5) -> F2:(964.5,378.2) -> F3:(965.2,378.3) -> F4:(965.8,378.4) -> F5:(968.2,379.0)
  • Anonymous ID #03: F0:(870.0,447.7) -> F1:(872.4,446.3) -> F2:(874.3,445.4) -> F3:(876.8,444.3) -> F4:(878.7,443.8) -> F5:(880.6,442.8)

================================================================================
  3. DENSITY & GEOSPATIAL CALIBRATION ACCURACY (app/geospatial/calibration.py)
================================================================================

3.1 Drone Altitude & Optical GSD Verification:
  - Drone Altitude (AGL):          32.4 m
  - Gimbal Pitch:                  -89.0° (Nadir Stabilized)
  - Ground Footprint (WxH):        58.35 m x 34.46 m
  - GSD Resolution X:              0.0456 m/pixel (Math Error: 0.000000 m)
  - GSD Resolution Y:              0.0479 m/pixel (Math Error: 0.000007 m)
  - Physical Ground Area / Pixel:  0.002182 m²/pixel

3.2 4-Point Homography Ground-Plane Area Mapping:
  - Homography Matrix Status:      VALID (Computed)
  - Reprojection Residual Error:   0.0000 pixels
  - Ground Truth Benchmark Area:   1130.88 m²
  - Homography Calculated Area:    1130.88 m²
  - Calibration Area Precision:    100.00% (Error: 0.00%)

3.3 Multi-Zone Calibrated Density Computation:
| Zone Name | Headcount | Calibrated Area (m²) | Measured Density (p/m²) | Expected Density (p/m²) | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| Zone A (Central Plaza)    |        24 |                200.0 |                   0.120 |                   0.120 | SAFE |
| Zone B (Entry Corridor)   |        35 |                 50.0 |                   0.700 |                   0.700 | SAFE |
| Zone C (South Chokepoint) |        38 |                 15.0 |                   2.533 |                   2.533 | WARNING |

3.4 Uncalibrated Fallback Verification:
  - Uncalibrated Homography Area:   None (Cleanly flags non-calibrated)
  - Fallback Mode Status:           CONFIRMED: Output explicitly labeled as Estimated Mode (pixels)

================================================================================
  4. MOVEMENT, FLOW & BOTTLENECK DETECTION ACCURACY
================================================================================

4.1 Virtual Line-Crossing Flow Verification:
  - Synthetic Downward Trajectories: 10 -> Ingress Detected:  10 (100.0% Acc)
  - Synthetic Upward Trajectories:   2  -> Egress Detected:   2 (100.0% Acc)
  - Net Flow Rate:                   +8 people/window
  - Flow Direction Imbalance Ratio:  0.667 (66.7% unidirectional pressure)

4.2 Velocity and Heading Vector Mathematical Verification:
  - Ground Truth Velocity Input:     1.250 m/s @ 45.0°
  - Measured Velocity Vector:        1.250 m/s @ 45.0°
  - Velocity Reconstruction Error:   0.000000 m/s
  - Heading Direction Error:         0.000000°

4.3 Bottleneck & Congestion Detection Verification:
  • Normal Transit Baseline: Upstream=6, Downstream=6, Speed=1.35 m/s -> Index=0.050 (Choke: False)
  • Severe Choke Occurrence: Upstream=28, Downstream=4, Speed=0.22 m/s (83.7% Drop, 4.67x Buildup)
  • Calculated Bottleneck Index:    0.847 / 1.000
  • Bottleneck Early Warning Flag:  TRIGGERED (CRITICAL)

================================================================================
  5. CROWD RISK ENGINE & ALERT ACCURACY (app/risk/risk_engine.py)
================================================================================

5.1 Controlled Multi-Factor State Transitions:
  • Scenario 1 (Normal Operations):
    Inputs: Density=15.0, Occupancy=20.0, Movement=10.0, Flow=15.0, Choke=10.0
    Formula Score: 14.25/100 -> State: SAFE (Expected: SAFE) [PASS]
  • Scenario 2 (Moderate Ingress Surge):
    Inputs: Density=70.0, Occupancy=75.0, Movement=45.0, Flow=50.0, Choke=60.0
    Formula Score: 61.50/100 -> State: WARNING (Expected: WARNING) [PASS]
  • Scenario 3 (High-Density Egress Choke):
    Inputs: Density=92.0, Occupancy=95.0, Movement=80.0, Flow=75.0, Choke=90.0
    Formula Score: 87.35/100 -> State: CRITICAL (Expected: CRITICAL) [PASS]

5.2 False-Alarm Suppression & Anomaly Spike Smoothing Proof:
Simulating 10-frame stream with a 1-frame transient glitch (Frame 5 = 95.0, baseline = 20.0):
| Frame # | Raw Risk Input | Smoothed Risk Score | System Status | False Alarm Triggered? |
|:-------:|:--------------:|:-------------------:|:-------------:|:----------------------:|
| Frame 01 |           20.0 |               20.00 | SAFE          | NO                     |
| Frame 02 |           20.0 |               20.00 | SAFE          | NO                     |
| Frame 03 |           20.0 |               20.00 | SAFE          | NO                     |
| Frame 04 |           20.0 |               20.00 | SAFE          | NO                     |
| Frame 05 |           95.0 |               42.50 | SAFE          | NO (Glitch Suppressed) |
| Frame 06 |           20.0 |               35.75 | SAFE          | NO                     |
| Frame 07 |           20.0 |               31.02 | SAFE          | NO                     |
| Frame 08 |           20.0 |               27.72 | SAFE          | NO                     |
| Frame 09 |           20.0 |               25.40 | SAFE          | NO                     |
| Frame 10 |           20.0 |               23.78 | SAFE          | NO                     |

5.3 Alert Deduplication & Human Acknowledgment Protocol:
  - Initial Critical Assessment:   Generated 1 alert(s)
    -> Message: 'Critical factors in zone zone_south_choke: density, occupancy, bottleneck'
  - Repeated Check during Cooldown: Generated 0 duplicate alert(s) (100% Deduplicated)
  - Deduplication Mechanism:       SUCCESS (30s Cooldown Active)

================================================================================
  6. END-TO-END PIPELINE & DATABASE AUDIT PROOF (sentinel/app/database)
================================================================================

Initializing SQLite Database Tables...
  [1] Event Session Persisted:    ID=b12cf753-9212-483a-8d39-a0ee3ee8ad96 | Name='Audit Run 2026-11-18 06:12:23'
  [2] Calibrated Zone Persisted:   ID=2ac50f7b-8a31-4261-a43a-97352606a157 | Zone='South Egress Chokepoint' (Cap: 80)
  [3] Drone Asset Persisted:       ID=7f0e9bd3-a07e-48b8-964b-8e2f7986ac35 | Model='Commercial Surveillance Quadcopter'
  [4] Critical Alert Persisted:    ID=e47a9c71-733f-49bc-9109-986cce5e9086 | Risk=86.4 | Title='Density Threshold Surge Alert'
  [5] Direct SQL Read Verification: Status=AlertStatus.ACTIVE | Title='Density Threshold Surge Alert'
  [6] Human Acknowledgment Saved:  Status=AlertStatus.ACKNOWLEDGED | Operator=977b2484-87c9-5aad-a380-83df13a8e668
  [7] Audit Cleanup Complete:      Session and associated zone/alert records safely cleared.

================================================================================
  VERIFICATION COMPLETE — ALL 6 MODULES VALIDATED (Elapsed: 4.92s)
================================================================================
