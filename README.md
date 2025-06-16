# Flit Packet Packing System

This project implements a flexible flit packing system with support for header/meta/data separation, granule-based packing, and visualization. It is highly configurable via JSON files and supports efficiency analysis and SVG output.

## Features

- **Configurable Flit Layout:** Define flit size, header positions, granule size, and number of granules in a JSON config.
- **Packet Types:** Support for multiple packet types with customizable header, meta, and data sizes.
- **Packing Algorithm:** Packs packets into flits and granules, supporting fragmentation and reporting statistics.
- **Efficiency Analysis:** Analyze packing efficiency over thousands of randomizations.
- **SVG Visualization:** Generates SVGs showing flit structure and actual packet packing.
- **CSV Export:** Optionally export all test run statistics to a CSV file.
- **Command-Line Interface:** Run as a script with arguments for config, output prefix, and CSV export.

## Usage

```bash
python flit_packer.py --config my_config.json
python flit_packer.py -c experiment.json -o exp1_
python flit_packer.py --csv-stats stats.csv
python flit_packer.py --help
```

### Arguments

- `--config`, `-c`: Path to configuration JSON file (default: `flit_config.json`)
- `--output-prefix`, `-o`: Prefix for output SVG files (default: none)
- `--csv-stats`: Output CSV file for test run statistics

## Configuration

Edit or create a JSON config file (see `flit_config.json` for an example):

```json
{
  "layout": {
    "flit_size": 256,
    "header_positions": [0, 64, 128, 192],
    "granule_size": 20,
    "num_granules": null
  },
  "packet_types": {
    "small_control": {
      "name": "Small Control",
      "header_size": 4,
      "meta_size": 2,
      "data_size": 10
    },
    ...
  },
  "test_packets": [
    {"type": "medium_data", "packet_id": 1},
    ...
  ],
  "test_distributions": [
    {
      "distribution": {"small_control": 10, "medium_data": 5, ...},
      "name": "Many Small Packets"
    },
    ...
  ]
}
```

## Output

- **SVG Files:** Visualizations of flit structure and example/test packings.
- **CSV File:** (Optional) All test run statistics for further analysis.

## Example

```bash
python flit_packer.py --config flit_config.json --output-prefix results_ --csv-stats all_stats.csv
```

This will generate SVG visualizations and a CSV file with statistics in the current directory.

## Requirements

- Python 3.7+
- No external dependencies (uses standard library)

## Customization

- Edit the config file to change flit layout, packet types, or test distributions.
- Add new packet types or distributions as needed.

## License

MIT License