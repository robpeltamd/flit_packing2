import os
import json
import sys
import re
from typing import Dict
from packet import Packet

def strip_json_comments(text: str) -> str:
    # Remove // line comments
    text = re.sub(r'//.*', '', text)
    # Remove /* block comments */
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text

def load_configuration(config_file: str = "flit_config.json") -> Dict:
    default_config = {
        "layout": {
            "flit_size": 256,
            "header_positions": [0, 64, 128, 192],
            "granule_size": 20,
            "num_granules": None
        },
        "packet_types": {
            "small_control": {
                "name": "Small Control",
                "header_size": 4,
                "meta_size": 2,
                "data_size": 10
            },
            "medium_data": {
                "name": "Medium Data",
                "header_size": 8,
                "meta_size": 4,
                "data_size": 20
            },
            "large_payload": {
                "name": "Large Payload",
                "header_size": 12,
                "meta_size": 6,
                "data_size": 46
            },
            "jumbo_transfer": {
                "name": "Jumbo Transfer",
                "header_size": 16,
                "meta_size": 8,
                "data_size": 104
            }
        },
        "test_packets": [
            {"type": "medium_data", "packet_id": 1},
            {"type": "large_payload", "packet_id": 2},
            {"type": "small_control", "packet_id": 3},
            {"type": "jumbo_transfer", "packet_id": 4},
            {"type": "medium_data", "packet_id": 5},
            {"type": "small_control", "packet_id": 6}
        ],
        "test_distributions": [
            {
                "distribution": {"small_control": 10, "medium_data": 5, "large_payload": 3, "jumbo_transfer": 1},
                "name": "Many Small Packets"
            },
            {
                "distribution": {"small_control": 2, "medium_data": 2, "large_payload": 2, "jumbo_transfer": 4},
                "name": "Many Large Packets"
            },
            {
                "distribution": {"small_control": 5, "medium_data": 5, "large_payload": 5, "jumbo_transfer": 5},
                "name": "Balanced Distribution"
            }
        ]
    }
    using_default_config = (config_file == "flit_config.json")
    if not os.path.exists(config_file):
        if using_default_config:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"Created default configuration file: {config_file}")
        else:
            print(f"Error: Configuration file '{config_file}' not found.")
            sys.exit(1)
    try:
        with open(config_file, 'r') as f:
            raw = f.read()
            raw = strip_json_comments(raw)
            config = json.loads(raw)
        print(f"Loaded configuration from: {config_file}")
        return config
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file '{config_file}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config file '{config_file}': {e}")
        sys.exit(1)

def create_packet_from_type(packet_type_name: str, packet_types: Dict, packet_id: int) -> Packet:
    if packet_type_name not in packet_types:
        raise ValueError(f"Unknown packet type: {packet_type_name}")
    packet_type = packet_types[packet_type_name]
    return Packet(
        size=0,
        packet_id=packet_id,
        name=packet_type["name"],
        header_size=packet_type["header_size"],
        meta_size=packet_type["meta_size"],
        data_size=packet_type["data_size"]
    )
