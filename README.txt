WAYFINDER — MAP MATCHING STUDIO (ML HACKATHON)

Open index.html directly in any web browser. The app runs completely client-side with zero external server dependencies.

FEATURES:
1. Multi-Trip Exploration:
   - Full support for all 10 competition trajectories (00000000, 00000007, 00000009, 00000040, 00000052, 00000056, 00000065, 00000076, 00000077, 00000091).
   - Selecting any trip immediately loads its authentic GPS observations, reference route, duration, distance, and bounding box.

2. Interactive Map with Smooth Pan & Zoom:
   - Click and drag anywhere on the map to pan.
   - Mouse wheel (scroll) to zoom in/out smoothly.
   - Zoom In / Zoom Out and Fit Route buttons.
   - Hardware-accelerated unified road path rendering 15,650 arcs with 60 FPS performance.
   - True metric aspect ratio corrected with cos(lat) factor.
   - Dynamic scale indicator (m / km) and real-time cursor coordinate tracker (lat, lon).

3. Client-Side ML Map Matching Engine:
   - Fast spatial grid hash index (<10ms build time).
   - Models available: LightGBM Scorer (Candidate), HMM + Viterbi (Baseline), and Ground Truth Reference.
   - "Run matching" performs orthogonal road projection, computes emission and transition metrics, updates the confidence ring gauge, and reports mean GPS offset (m), matched percentage, sequence similarity, and inference latency.
   - Toggleable "GPS Error Vectors" layer showing connector lines between raw GPS observations and matched road points.

4. Real-Time GPS Playback & Vehicle Simulation:
   - Scrubbable timeline with live time counter.
   - Play/Pause with speed multipliers (1x, 2x, 5x).
   - Animated vehicle marker with directional bearing indicator.
   - Real-time telemetry HUD displaying speed (km/h), heading (deg), and GPS offset (m).

5. Data Import & Export:
   - "Import data" button and Drag-and-Drop file import directly onto the map (supports .track, .csv, .tsv, .txt).
   - "Export route" button to export the predicted route (.route format) for competition submission.

6. Extended Workspaces:
   - Models: Architecture matrix and live hyperparameter tuner (sigma_gps, beta, search radius, heading weight).
   - Evaluation: Full validation benchmark suite with live progress runner and JSON report exporter.
   - Trip Explorer: Comprehensive trajectory table with 1-click "Explore on Map".
   - Datasets: File format documentation (.track, .nodes, .arcs, .route) and live record inspector.
