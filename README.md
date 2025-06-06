# Flit Packing System

This Python module implements a system for packing byte packets into granules and flits, visualizing the packing structure, and analyzing packing efficiency. The system is designed to handle various packet sizes and distributions, providing insights into packing efficiency and fragmentation.

## Features

- **Packet Packing**: Pack packets into granules and flits based on a configurable layout.
- **Visualization**: Generate SVG files to visualize the flit structure and packed flits.
- **Efficiency Analysis**: Analyze packing efficiency over multiple randomizations of packet distributions.

## Installation

1. Clone the repository or copy the files to your local machine.
2. Ensure you have Python 3.6 or higher installed.
3. Install any required dependencies:
   ```bash
   pip install -r requirements.txt

(Note: If requirements.txt is not provided, ensure the standard library modules like dataclasses, xml.etree.ElementTree, and statistics are available.)

Usage
Running the Module
Open a terminal and navigate to the directory containing flit_packer.py.
Run the script:
The script will:
Load the configuration from ualink_over_c2c.json.
Generate visualizations (flit_structure.svg, example_packing.svg, and test SVG files).
Print packing statistics and efficiency analysis results to the terminal.
Customizing Configuration
Edit the ualink_over_c2c.json file to modify the layout, packet sizes, and test distributions.
Re-run the script to apply the changes.
Viewing Visualizations
Open the generated SVG files (flit_structure.svg, example_packing.svg, etc.) in a web browser or an SVG viewer to explore the flit structure and packing results.

Example Output
After running the script, you will see output like:
Flit Packing System with Granules
==================================================
Configuration loaded from: ualink_over_c2c.json
Flit size: 256 bytes
Header positions: [0, 64, 128, 192]
Granule size: 20 bytes
Granules per flit: 12
Available data regions: [(1, 64), (65, 128), (129, 192), (193, 256)]
Total data capacity per flit: 240 bytes
Data efficiency: 93.8%
Granule capacity per flit: 240 bytes

Example Packing Results:
Total packets: 4
Total packet bytes: 688
Flits generated: 3
Full flits analyzed for efficiency: 2
Granules generated: 36
Flit efficiency (full flits only): 91.67%
Granule efficiency (full flits only): 91.67%
Fragmented packets: 1
Total waste bytes: 32
Wasted granule bytes: 32

Generated Files
flit_structure.svg: Basic flit layout showing headers and granule regions.
example_packing.svg: Example packet packing demonstration.
test_1_packing.svg: Visualization for "Many Small Packets" test distribution.
test_2_packing.svg: Visualization for "Many Large Packets" test distribution.
test_3_packing.svg: Visualization for "Balanced Distribution" test distribution.
License
This module is provided under the MIT License. Feel free to use, modify, and distribute it.

Support
For questions or issues, please contact the author or open an issue in the repository.