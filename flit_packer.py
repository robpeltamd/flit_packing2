# flit_packer.py - MAIN FILE
#!/usr/bin/env python3
"""
Flit Packet Packing System with Header-Meta-Data Support

This is the main file for the flit packing system.
It imports all the necessary classes and provides the main entry point.

Usage:
    python flit_packer.py --config my_config.json
    python flit_packer.py -c experiment.json -o exp1_
    python flit_packer.py --help
"""

import random
import statistics
import argparse
import sys
import json
import os
import csv
from typing import List, Tuple, Dict
import xml.etree.ElementTree as ET

from packet import Packet
from granule import Granule
from flit_layout import FlitLayout
from utils import load_configuration, create_packet_from_type

class FlitPacker:
    """Handles packing of packets into granules and flits with header/meta/data separation"""
    
    def __init__(self, layout: FlitLayout, packet_sizes: List[int]):
        self.layout = layout
        self.packet_sizes = packet_sizes
        self.available_regions = layout.get_available_positions()
        self.granule_layout = layout.get_granule_layout()
        self.total_data_capacity = layout.data_capacity
        self.granules_per_flit = layout.num_granules
        self.granule_size = layout.granule_size
    
    def _start_new_flit(self) -> Tuple[bytearray, List[Granule]]:
        """Start a new flit and return flit bytearray and granules list"""
        new_flit = bytearray(self.layout.flit_size)
        
        # Initialize flit headers (using 0xFF as header marker)
        for pos in self.layout.header_positions:
            new_flit[pos] = 0xFF
        
        # Create new granules (all start as data type)
        new_granules = []
        for i in range(self.granules_per_flit):
            granule = Granule(
                granule_id=i,
                size=self.granule_size,
                flit_id=0,  # Will be updated when added to flits list
                granule_type="data"
            )
            new_granules.append(granule)
        
        return new_flit, new_granules
    
    def _finish_current_flit(self, current_flit: bytearray, current_granules: List[Granule], 
                            flits: List[bytes], all_granules: List[List[Granule]]):
        """Finish current flit and add to collections"""
        self._place_granules_in_flit(current_flit, current_granules)
        flits.append(bytes(current_flit))
        all_granules.append(current_granules)
    
    def pack_packets(self, packets: List[Packet]) -> Tuple[List[bytes], List[List[Granule]], Dict]:
        """Pack packets into granules and flits with header/meta/data separation"""
        flits = []
        all_granules = []
        
        # Create initial set of granules for first flit
        current_flit, current_granules = self._start_new_flit()
        
        total_packet_bytes = sum(p.size for p in packets)
        packed_bytes = 0
        fragmented_packets = 0
        wasted_granule_bytes = 0
        
        stats = {
            'total_flits': 0,
            'total_granules': 0,
            'total_packet_bytes': total_packet_bytes,
            'packed_bytes': 0,
            'fragmented_packets': 0,
            'efficiency': 0.0,
            'waste_bytes': 0,
            'granule_efficiency': 0.0,
            'wasted_granule_bytes': 0,
            'full_flits_analyzed': 0
        }
        
        for packet in packets:
            # Pack header, meta, data (must be in consecutive granules)
            remaining_header_size = packet.header_size if hasattr(packet, 'header_size') else 0
            remaining_meta_size = packet.meta_size if hasattr(packet, 'meta_size') else 0
            remaining_data_size = packet.data_size if hasattr(packet, 'data_size') else packet.size
            packet_fragmented = False
            
            # Find available consecutive granules (header + meta + data)
            header_granule_idx = None
            meta_granule_idx = None
            data_granule_idx = None
            
            # Determine how many consecutive granules we need
            granules_needed = 0
            if remaining_header_size > 0:
                granules_needed += 1
            if remaining_meta_size > 0:
                granules_needed += 1
            if remaining_data_size > 0:
                granules_needed += 1
            
            # Look for available consecutive granules
            if granules_needed > 0:
                for i in range(len(current_granules) - granules_needed + 1):
                    # Check if we have enough consecutive available granules
                    available_consecutive = True
                    for j in range(granules_needed):
                        if current_granules[i + j].packet_id is not None:
                            available_consecutive = False
                            break
                    
                    if available_consecutive:
                        # Check size constraints
                        valid_sizes = True
                        granule_idx = i
                        
                        if remaining_header_size > 0:
                            if current_granules[granule_idx].size < remaining_header_size:
                                valid_sizes = False
                            else:
                                header_granule_idx = granule_idx
                                granule_idx += 1
                        
                        if remaining_meta_size > 0 and valid_sizes:
                            if current_granules[granule_idx].size < remaining_meta_size:
                                valid_sizes = False
                            else:
                                meta_granule_idx = granule_idx
                                granule_idx += 1
                        
                        if remaining_data_size > 0 and valid_sizes:
                            data_granule_idx = granule_idx
                        
                        if valid_sizes:
                            # Set granule types
                            if header_granule_idx is not None:
                                current_granules[header_granule_idx].granule_type = "header"
                            if meta_granule_idx is not None:
                                current_granules[meta_granule_idx].granule_type = "meta"
                            if data_granule_idx is not None:
                                current_granules[data_granule_idx].granule_type = "data"
                            break
                        else:
                            # Reset for next iteration
                            header_granule_idx = None
                            meta_granule_idx = None
                            data_granule_idx = None
                
                # If no consecutive granules found, need new flit
                if (remaining_header_size > 0 and header_granule_idx is None) or \
                   (remaining_meta_size > 0 and meta_granule_idx is None) or \
                   (remaining_data_size > 0 and data_granule_idx is None):
                    
                    self._finish_current_flit(current_flit, current_granules, flits, all_granules)
                    wasted_granule_bytes += sum(g.size - g.bytes_used for g in current_granules)
                    
                    # Start new flit
                    current_flit, current_granules = self._start_new_flit()
                    
                    # Use first granules for header + meta + data
                    if len(current_granules) >= granules_needed:
                        granule_idx = 0
                        if remaining_header_size > 0:
                            current_granules[granule_idx].granule_type = "header"
                            header_granule_idx = granule_idx
                            granule_idx += 1
                        if remaining_meta_size > 0:
                            current_granules[granule_idx].granule_type = "meta"
                            meta_granule_idx = granule_idx
                            granule_idx += 1
                        if remaining_data_size > 0:
                            current_granules[granule_idx].granule_type = "data"
                            data_granule_idx = granule_idx
                        
                        packet_fragmented = True
            
            # Pack header if present
            if remaining_header_size > 0 and header_granule_idx is not None:
                header_granule = current_granules[header_granule_idx]
                bytes_added = header_granule.add_packet_data(packet, remaining_header_size, "header")
                remaining_header_size -= bytes_added
                packed_bytes += bytes_added
                
                if remaining_header_size > 0:
                    print(f"Warning: Packet {packet.packet_id} header ({packet.header_size}B) too large for granule ({self.granule_size}B)")
            
            # Pack metadata if present
            if remaining_meta_size > 0 and meta_granule_idx is not None:
                meta_granule = current_granules[meta_granule_idx]
                bytes_added = meta_granule.add_packet_data(packet, remaining_meta_size, "meta")
                remaining_meta_size -= bytes_added
                packed_bytes += bytes_added
                
                if remaining_meta_size > 0:
                    print(f"Warning: Packet {packet.packet_id} metadata ({packet.meta_size}B) too large for granule ({self.granule_size}B)")
            
            # Pack data
            while remaining_data_size > 0:
                if data_granule_idx is not None and data_granule_idx < len(current_granules):
                    data_granule = current_granules[data_granule_idx]
                    bytes_added = data_granule.add_packet_data(packet, remaining_data_size, "data")
                    remaining_data_size -= bytes_added
                    packed_bytes += bytes_added
                
                # If more data remains, look for next available data granule
                if remaining_data_size > 0:
                    next_data_granule_idx = None
                    for i in range(data_granule_idx + 1, len(current_granules)):
                        granule = current_granules[i]
                        if granule.packet_id is None:
                            granule.granule_type = "data"
                            next_data_granule_idx = i
                            break
                    
                    if next_data_granule_idx is not None:
                        data_granule_idx = next_data_granule_idx
                    else:
                        # Need new flit for remaining data
                        self._finish_current_flit(current_flit, current_granules, flits, all_granules)
                        wasted_granule_bytes += sum(g.size - g.bytes_used for g in current_granules)
                        
                        current_flit, current_granules = self._start_new_flit()
                        current_granules[0].granule_type = "data"
                        data_granule_idx = 0
                        packet_fragmented = True
            
            if packet_fragmented:
                fragmented_packets += 1
        
        # Handle the last flit if it has data
        if any(g.bytes_used > 0 for g in current_granules):
            for granule in current_granules:
                wasted_granule_bytes += granule.size - granule.bytes_used
            
            self._finish_current_flit(current_flit, current_granules, flits, all_granules)
        
        # Calculate statistics
        stats['total_flits'] = len(flits)
        stats['total_granules'] = len(flits) * self.granules_per_flit
        stats['packed_bytes'] = packed_bytes
        stats['fragmented_packets'] = fragmented_packets
        stats['wasted_granule_bytes'] = wasted_granule_bytes
        
        # For efficiency calculations, exclude the last flit as it's always partially filled
        flits_for_efficiency = max(0, len(flits) - 1) if len(flits) > 1 else 0
        granules_for_efficiency = flits_for_efficiency * self.granules_per_flit
        
        # Calculate capacities for efficiency metrics
        total_data_capacity_for_efficiency = flits_for_efficiency * self.total_data_capacity
        total_granule_capacity_for_efficiency = granules_for_efficiency * self.granule_size
        total_flit_capacity_for_efficiency = flits_for_efficiency * self.layout.flit_size
        
        # Calculate efficiency based on full flits only
        if flits_for_efficiency > 0:
            if len(flits) > 1:
                last_flit_granules = all_granules[-1] if all_granules else []
                last_flit_packed_bytes = sum(g.bytes_used for g in last_flit_granules)
                packed_bytes_full_flits = packed_bytes - last_flit_packed_bytes
                
                header_bytes_full_flits = flits_for_efficiency * len(self.layout.header_positions)
                
                stats['data_efficiency'] = (packed_bytes_full_flits / total_data_capacity_for_efficiency * 100) if total_data_capacity_for_efficiency > 0 else 0.0
                stats['granule_efficiency'] = (packed_bytes_full_flits / total_granule_capacity_for_efficiency * 100) if total_granule_capacity_for_efficiency > 0 else 0.0
                stats['overall_efficiency'] = ((packed_bytes_full_flits + header_bytes_full_flits) / total_flit_capacity_for_efficiency * 100) if total_flit_capacity_for_efficiency > 0 else 0.0
                stats['efficiency'] = stats['data_efficiency']
            else:
                header_bytes_single_flit = len(self.layout.header_positions)
                
                stats['data_efficiency'] = (packed_bytes / self.total_data_capacity * 100) if self.total_data_capacity > 0 else 0.0
                stats['granule_efficiency'] = (packed_bytes / (self.granules_per_flit * self.granule_size) * 100) if (self.granules_per_flit * self.granule_size) > 0 else 0.0
                stats['overall_efficiency'] = ((packed_bytes + header_bytes_single_flit) / self.layout.flit_size * 100) if self.layout.flit_size > 0 else 0.0
                stats['efficiency'] = stats['data_efficiency']
        else:
            if len(flits) == 1:
                header_bytes_single_flit = len(self.layout.header_positions)
                
                stats['data_efficiency'] = (packed_bytes / self.total_data_capacity * 100) if self.total_data_capacity > 0 else 0.0
                stats['granule_efficiency'] = (packed_bytes / (self.granules_per_flit * self.granule_size) * 100) if (self.granules_per_flit * self.granule_size) > 0 else 0.0
                stats['overall_efficiency'] = ((packed_bytes + header_bytes_single_flit) / self.layout.flit_size * 100) if self.layout.flit_size > 0 else 0.0
                stats['efficiency'] = stats['data_efficiency']
            else:
                stats['data_efficiency'] = 0.0
                stats['granule_efficiency'] = 0.0
                stats['overall_efficiency'] = 0.0
                stats['efficiency'] = 0.0
        
        total_capacity = len(flits) * self.total_data_capacity
        stats['waste_bytes'] = total_capacity - packed_bytes
        stats['full_flits_analyzed'] = flits_for_efficiency
        
        return flits, all_granules, stats
    
    def _place_granules_in_flit(self, flit: bytearray, granules: List[Granule]):
        """Place granule data into flit at appropriate positions"""
        granule_positions = self.layout.get_granule_layout()
        
        for granule_idx, granule in enumerate(granules):
            if granule_idx >= self.granules_per_flit:
                break
            
            granule_byte_positions = [pos for pos in granule_positions if pos[2] == granule_idx]
            
            data_idx = 0
            for start_pos, end_pos, _ in granule_byte_positions:
                bytes_to_copy = min(end_pos - start_pos, granule.bytes_used - data_idx)
                if bytes_to_copy <= 0:
                    break
                
                for i in range(bytes_to_copy):
                    flit[start_pos + i] = granule.data[data_idx + i]
                
                data_idx += bytes_to_copy

class PackingAnalyzer:
    """Analyzes packing efficiency over multiple randomizations"""
    
    def __init__(self, packer: FlitPacker, packet_types: Dict = None):
        self.packer = packer
        self.packet_types = packet_types or {}
    
    def analyze_efficiency(self, packet_counts: Dict[int, int], num_iterations: int = 1000) -> Dict:
        """Analyze packing efficiency over multiple random packet orderings"""
        efficiencies = []
        granule_efficiencies = []
        overall_efficiencies = []
        flit_counts = []
        fragmentation_rates = []
        
        for _ in range(num_iterations):
            # Generate random packet sequence using proper packet types
            packets = []
            packet_id = 0
            
            for size, count in packet_counts.items():
                for _ in range(count):
                    # Find the packet type that matches this size
                    packet_type_info = None
                    for type_name, type_data in self.packet_types.items():
                        if type_data["header_size"] + type_data["meta_size"] + type_data["data_size"] == size:
                            packet_type_info = type_data
                            break
                    
                    if packet_type_info:
                        # Create packet with header/meta/data info
                        packet = Packet(
                            size=size,
                            packet_id=packet_id,
                            name=packet_type_info["name"],
                            header_size=packet_type_info["header_size"],
                            meta_size=packet_type_info["meta_size"],
                            data_size=packet_type_info["data_size"]
                        )
                    else:
                        # Fallback to basic packet
                        packet = Packet(size, packet_id)
                    
                    packets.append(packet)
                    packet_id += 1
            
            # Randomize order
            random.shuffle(packets)
            
            # Pack and analyze
            flits, granules, stats = self.packer.pack_packets(packets)
            
            efficiencies.append(stats['data_efficiency'])
            granule_efficiencies.append(stats['granule_efficiency'])
            overall_efficiencies.append(stats['overall_efficiency'])
            flit_counts.append(stats['total_flits'])
            
            total_packets = len(packets)
            fragmentation_rate = (stats['fragmented_packets'] / total_packets * 100) if total_packets > 0 else 0
            fragmentation_rates.append(fragmentation_rate)
        
        return {
            'efficiency_mean': statistics.mean(efficiencies),
            'efficiency_std': statistics.stdev(efficiencies) if len(efficiencies) > 1 else 0,
            'efficiency_min': min(efficiencies),
            'efficiency_max': max(efficiencies),
            'granule_efficiency_mean': statistics.mean(granule_efficiencies),
            'granule_efficiency_std': statistics.stdev(granule_efficiencies) if len(granule_efficiencies) > 1 else 0,
            'overall_efficiency_mean': statistics.mean(overall_efficiencies),
            'overall_efficiency_std': statistics.stdev(overall_efficiencies) if len(overall_efficiencies) > 1 else 0,
            'overall_efficiency_min': min(overall_efficiencies),
            'overall_efficiency_max': max(overall_efficiencies),
            'flit_count_mean': statistics.mean(flit_counts),
            'flit_count_std': statistics.stdev(flit_counts) if len(flit_counts) > 1 else 0,
            'fragmentation_rate_mean': statistics.mean(fragmentation_rates),
            'fragmentation_rate_std': statistics.stdev(fragmentation_rates) if len(fragmentation_rates) > 1 else 0,
            'iterations': num_iterations
        }

class FlitVisualizer:
    """Creates SVG visualization of flit format with granules"""
    
    def __init__(self, layout: FlitLayout):
        self.layout = layout
        # Generate colors for different packet IDs
        self.packet_colors = [
            '#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#f0932b', 
            '#eb4d4b', '#6c5ce7', '#a29bfe', '#fd79a8', '#fdcb6e',
            '#00b894', '#00cec9', '#74b9ff', '#0984e3', '#6c5ce7'
        ]
    
    def create_structure_svg(self, width: int = 2000, height: int = 400) -> str:
        """Create SVG visualization of flit structure with granules"""
        
        # Calculate dimensions - 64 bytes per row
        bytes_per_row = 64
        cell_width = 25
        cell_height = 20
        margin = 50
        
        # Create SVG root
        svg = ET.Element('svg', {
            'width': str(width),
            'height': str(height),
            'xmlns': 'http://www.w3.org/2000/svg'
        })
        
        # Add styles
        style = ET.SubElement(svg, 'style')
        granule_colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#f39c12', '#34495e', '#e74c3c', '#95a5a6']
        style_text = """
            .header { fill: #2c3e50; stroke: #000; stroke-width: 2; }
            .empty { fill: #ecf0f1; stroke: #bdc3c7; stroke-width: 1; }
            .granule { stroke: #e74c3c; stroke-width: 2; fill: none; stroke-dasharray: 3,3; }
            .text { font-family: Arial, sans-serif; font-size: 6px; text-anchor: middle; }
            .title { font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-anchor: middle; }
            .legend { font-family: Arial, sans-serif; font-size: 12px; }
            .granule-text { font-family: Arial, sans-serif; font-size: 8px; text-anchor: middle; fill: #e74c3c; font-weight: bold; }
        """
        
        for i, color in enumerate(granule_colors):
            style_text += f"\n            .granule{i} {{ fill: {color}; stroke: #000; stroke-width: 1; opacity: 0.7; }}"
        
        style.text = style_text
        
        # Title
        title = ET.SubElement(svg, 'text', {
            'x': str(width // 2),
            'y': '30',
            'class': 'title'
        })
        title.text = f'Flit Structure with Granules ({self.layout.flit_size} bytes)'
        
        # Get granule layout
        granule_positions = self.layout.get_granule_layout()
        byte_to_granule = {}
        for start_pos, end_pos, granule_id in granule_positions:
            for byte_pos in range(start_pos, end_pos):
                byte_to_granule[byte_pos] = granule_id
        
        # Draw byte grid
        start_x = margin
        start_y = 60
        
        for i in range(self.layout.flit_size):
            row = i // bytes_per_row
            col = i % bytes_per_row
            
            x = start_x + col * cell_width
            y = start_y + row * cell_height
            
            # Determine cell type and color
            if i in self.layout.header_positions:
                cell_class = 'header'
                cell_text = 'H'
            elif i in byte_to_granule:
                granule_id = byte_to_granule[i]
                cell_class = f'granule{granule_id % len(granule_colors)}'
                cell_text = str(granule_id)
            else:
                cell_class = 'empty'
                cell_text = ''
            
            # Draw cell
            rect = ET.SubElement(svg, 'rect', {
                'x': str(x),
                'y': str(y),
                'width': str(cell_width - 1),
                'height': str(cell_height - 1),
                'class': cell_class
            })
            
            # Add byte position number
            pos_text = ET.SubElement(svg, 'text', {
                'x': str(x + cell_width // 2),
                'y': str(y + 8),
                'class': 'text'
            })
            pos_text.text = str(i)
            
            # Add granule ID or header marker
            if cell_text:
                content_text = ET.SubElement(svg, 'text', {
                    'x': str(x + cell_width // 2),
                    'y': str(y + cell_height - 5),
                    'class': 'granule-text' if cell_class.startswith('granule') else 'text'
                })
                content_text.text = cell_text
        
        # Add configuration info
        rows = (self.layout.flit_size + bytes_per_row - 1) // bytes_per_row
        info_y = start_y + (rows + 1) * cell_height + 30
        
        config_items = [
            f'Flit size: {self.layout.flit_size} bytes',
            f'Header positions: {self.layout.header_positions}',
            f'Granule size: {self.layout.granule_size} bytes',
            f'Granules per flit: {self.layout.num_granules}',
            f'Data capacity: {self.layout.data_capacity} bytes ({self.layout.data_capacity/self.layout.flit_size*100:.1f}%)'
        ]
        
        for i, item in enumerate(config_items):
            config_text = ET.SubElement(svg, 'text', {
                'x': str(start_x),
                'y': str(info_y + i * 18),
                'class': 'legend'
            })
            config_text.text = item
        
        # Add bottom padding
        info_y += 40
        
        return ET.tostring(svg, encoding='unicode')
    
    def create_packed_flits_svg(self, flits: List[bytes], granules_list: List[List[Granule]], 
                               packets: List[Packet], stats: Dict, test_name: str = "", 
                               width: int = 2000, height: int = None) -> str:
        """Create SVG visualization showing actual packet packing in granules and flits"""
        
        bytes_per_row = 64
        cell_width = 20
        cell_height = 15
        margin = 50
        flit_spacing = 40
        title_height = 100
        
        # Calculate height based on number of flits
        if height is None:
            rows_per_flit = (self.layout.flit_size + bytes_per_row - 1) // bytes_per_row
            height = title_height + len(flits) * (rows_per_flit * cell_height + flit_spacing) + 400  # Added more padding
        
        # Create SVG root
        svg = ET.Element('svg', {
            'width': str(width),
            'height': str(height),
            'xmlns': 'http://www.w3.org/2000/svg'
        })
        
        # Add styles
        style = ET.SubElement(svg, 'style')
        style_text = """
            .header { fill: #2c3e50; stroke: #000; stroke-width: 1; }
            .empty { fill: #ecf0f1; stroke: #bdc3c7; stroke-width: 1; }
            .unused-granule { fill: #f39c12; stroke: #000; stroke-width: 1; opacity: 0.6; }
            .granule-border { stroke: #e74c3c; stroke-width: 2; fill: none; }
            .text { font-family: Arial, sans-serif; font-size: 5px; text-anchor: middle; }
            .title { font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-anchor: middle; }
            .flit-title { font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; }
            .legend { font-family: Arial, sans-serif; font-size: 11px; }
            .stats { font-family: Arial, sans-serif; font-size: 12px; }
            .packet-text { font-family: Arial, sans-serif; font-size: 5px; text-anchor: middle; fill: white; font-weight: bold; }
            .unused-text { font-family: Arial, sans-serif; font-size: 5px; text-anchor: middle; fill: #d35400; font-weight: bold; }
        """
        
        # Add packet-specific colors to styles
        for i, color in enumerate(self.packet_colors):
            style_text += f"\n            .packet{i} {{ fill: {color}; stroke: #000; stroke-width: 1; }}"
        
        style.text = style_text
        
        # Title
        title = ET.SubElement(svg, 'text', {
            'x': str(width // 2),
            'y': '25',
            'class': 'title'
        })
        title.text = f'Granule-Based Packet Packing - {test_name}'
        
        # Stats
        stats_y = 50
        stats_text = ET.SubElement(svg, 'text', {
            'x': str(width // 2),
            'y': str(stats_y),
            'class': 'stats',
            'text-anchor': 'middle'
        })
        stats_text.text = f"Data Eff: {stats['data_efficiency']:.1f}% | Granule Eff: {stats['granule_efficiency']:.1f}% | Overall Eff: {stats['overall_efficiency']:.1f}% | Flits: {stats['total_flits']} ({stats.get('full_flits_analyzed', 0)} analyzed) | Fragmented: {stats['fragmented_packets']} packets"
        
        # Configuration info
        config_y = stats_y + 20
        config_text = ET.SubElement(svg, 'text', {
            'x': str(width // 2),
            'y': str(config_y),
            'class': 'legend',
            'text-anchor': 'middle'
        })
        config_text.text = f"Granule size: {self.layout.granule_size}B | Granules per flit: {self.layout.num_granules} | Headers: {len(self.layout.header_positions)}"
        
        # Draw each flit
        current_y = title_height
        
        for flit_idx, (flit_data, flit_granules) in enumerate(zip(flits, granules_list)):
            # Flit title with granule usage
            used_granules = sum(1 for g in flit_granules if g.bytes_used > 0)
            total_used_bytes = sum(g.bytes_used for g in flit_granules)
            total_granule_capacity = len(flit_granules) * self.layout.granule_size
            wasted_granule_bytes = total_granule_capacity - total_used_bytes
            
            flit_title = ET.SubElement(svg, 'text', {
                'x': str(margin),
                'y': str(current_y),
                'class': 'flit-title'
            })
            flit_title.text = f'Flit {flit_idx + 1}'
            
            granule_summary = ET.SubElement(svg, 'text', {
                'x': str(margin + 120),
                'y': str(current_y),
                'class': 'legend'
            })
            granule_summary.text = f'({used_granules}/{len(flit_granules)} granules, {wasted_granule_bytes}B wasted)'
            
            current_y += 25
            
            # Get granule layout and create status mapping
            granule_positions = self.layout.get_granule_layout()
            byte_to_granule = {}
            for start_pos, end_pos, granule_id in granule_positions:
                for byte_pos in range(start_pos, end_pos):
                    byte_to_granule[byte_pos] = granule_id
            
            # Create granule usage mapping
            granule_usage = {}
            granule_packets = {}
            granule_types = {}
            for granule in flit_granules:
                granule_usage[granule.granule_id] = granule.bytes_used
                granule_types[granule.granule_id] = granule.granule_type
                if granule.packet_id is not None:
                    granule_packets[granule.granule_id] = granule.packet_id
            
            # Create byte-level status mapping
            byte_to_granule_status = {}
            for start_pos, end_pos, granule_id in granule_positions:
                bytes_used_in_granule = granule_usage.get(granule_id, 0)
                for i, byte_pos in enumerate(range(start_pos, end_pos)):
                    # Calculate which byte within the granule this represents
                    granule_byte_index = 0
                    # Find how many bytes of this granule have been placed before this segment
                    for prev_start, prev_end, prev_gid in granule_positions:
                        if prev_gid == granule_id and prev_start < start_pos:
                            granule_byte_index += prev_end - prev_start
                    granule_byte_index += i
                    
                    if granule_byte_index < bytes_used_in_granule:
                        byte_to_granule_status[byte_pos] = ('used', granule_id)
                    else:
                        byte_to_granule_status[byte_pos] = ('unused', granule_id)
            
            # Draw flit grid
            for byte_idx in range(self.layout.flit_size):
                row = byte_idx // bytes_per_row
                col = byte_idx % bytes_per_row
                
                x = margin + col * cell_width
                y = current_y + row * cell_height
                
                # Determine cell type and color
                if byte_idx in self.layout.header_positions:
                    cell_class = 'header'
                    cell_text = 'H'
                elif byte_idx in byte_to_granule_status:
                    status, granule_id = byte_to_granule_status[byte_idx]
                    granule_type = granule_types.get(granule_id, "data")
                    
                    if status == 'used' and granule_id in granule_packets:
                        packet_id = granule_packets[granule_id]
                        if granule_type == "header":
                            cell_class = f'packet{packet_id % len(self.packet_colors)}'
                            cell_text = f'H{packet_id}'
                        elif granule_type == "meta":
                            cell_class = f'packet{packet_id % len(self.packet_colors)}'
                            cell_text = f'M{packet_id}'
                        else:
                            cell_class = f'packet{packet_id % len(self.packet_colors)}'
                            cell_text = str(packet_id)
                    elif status == 'unused':
                        cell_class = 'unused-granule'
                        cell_text = 'U'
                    else:
                        cell_class = 'empty'
                        if granule_type == "header":
                            cell_text = f'H{granule_id}'
                        elif granule_type == "meta":
                            cell_text = f'M{granule_id}'
                        else:
                            cell_text = str(granule_id)
                elif byte_idx in byte_to_granule:
                    granule_id = byte_to_granule[byte_idx]
                    cell_class = 'empty'
                    cell_text = str(granule_id)
                else:
                    cell_class = 'empty'
                    cell_text = ''
                
                # Draw cell
                rect = ET.SubElement(svg, 'rect', {
                    'x': str(x),
                    'y': str(y),
                    'width': str(cell_width - 1),
                    'height': str(cell_height - 1),
                    'class': cell_class
                })
                
                # Add text
                if cell_text:
                    text_class = 'packet-text' if cell_class.startswith('packet') else 'unused-text' if cell_class == 'unused-granule' else 'text'
                    text = ET.SubElement(svg, 'text', {
                        'x': str(x + cell_width // 2),
                        'y': str(y + cell_height // 2 + 2),
                        'class': text_class
                    })
                    text.text = cell_text
            
            rows_in_flit = (self.layout.flit_size + bytes_per_row - 1) // bytes_per_row
            current_y += rows_in_flit * cell_height + flit_spacing
        
        # Add packet legend
        legend_y = current_y + 20
        legend_title = ET.SubElement(svg, 'text', {
            'x': str(margin),
            'y': str(legend_y),
            'class': 'flit-title'
        })
        legend_title.text = 'Legend:'
        
        legend_y += 20
        
        # Header legend
        header_rect = ET.SubElement(svg, 'rect', {
            'x': str(margin),
            'y': str(legend_y - 8),
            'width': '12',
            'height': '10',
            'class': 'header'
        })
        header_text = ET.SubElement(svg, 'text', {
            'x': str(margin + 18),
            'y': str(legend_y),
            'class': 'legend'
        })
        header_text.text = 'Headers'
        
        # Header granule legend
        header_granule_rect = ET.SubElement(svg, 'rect', {
            'x': str(margin + 100),
            'y': str(legend_y - 8),
            'width': '12',
            'height': '10',
            'class': 'packet0'
        })
        header_granule_text = ET.SubElement(svg, 'text', {
            'x': str(margin + 118),
            'y': str(legend_y),
            'class': 'legend'
        })
        header_granule_text.text = 'Header granules (H#)'
        
        # Unused granule space legend
        legend_y += 20
        unused_rect = ET.SubElement(svg, 'rect', {
            'x': str(margin),
            'y': str(legend_y - 8),
            'width': '12',
            'height': '10',
            'class': 'unused-granule'
        })
        unused_text = ET.SubElement(svg, 'text', {
            'x': str(margin + 18),
            'y': str(legend_y),
            'class': 'legend'
        })
        unused_text.text = 'Unused granule space'
        
        legend_y += 25
        
        # Packet legend
        packet_legend_title = ET.SubElement(svg, 'text', {
            'x': str(margin),
            'y': str(legend_y),
            'class': 'legend'
        })
        packet_legend_title.text = 'Packets:'
        
        legend_y += 15
        col_count = 0
        for packet in packets:
            if col_count >= 10:  # Start new row after 10 items
                legend_y += 20
                col_count = 0
            
            x_offset = margin + col_count * 130
            
            # Color box
            color_class = f'packet{packet.packet_id % len(self.packet_colors)}'
            rect = ET.SubElement(svg, 'rect', {
                'x': str(x_offset),
                'y': str(legend_y - 8),
                'width': '12',
                'height': '10',
                'class': color_class
            })
            
            # Packet info
            packet_text = ET.SubElement(svg, 'text', {
                'x': str(x_offset + 18),
                'y': str(legend_y),
                'class': 'legend'
            })
            if hasattr(packet, 'name') and packet.name:
                packet_text.text = f'P{packet.packet_id}: {packet.name} ({packet.size}B)'
            else:
                packet_text.text = f'P{packet.packet_id}({packet.size}B)'
            
            col_count += 1
        # Meta granule legend
        meta_granule_rect = ET.SubElement(svg, 'rect', {
            'x': str(margin + 250),
            'y': str(legend_y - 8),
            'width': '12',
            'height': '10',
            'class': 'packet1'
        })
        meta_granule_text = ET.SubElement(svg, 'text', {
            'x': str(margin + 268),
            'y': str(legend_y),
            'class': 'legend'
        })
        meta_granule_text.text = 'Meta granules (M#)'
        
        # Unused granule space legend
        legend_y += 20
        unused_rect = ET.SubElement(svg, 'rect', {
            'x': str(margin),
            'y': str(legend_y - 8),
            'width': '12',
            'height': '10',
            'class': 'unused-granule'
        })
        unused_text = ET.SubElement(svg, 'text', {
            'x': str(margin + 18),
            'y': str(legend_y),
            'class': 'legend'
        })
        unused_text.text = 'Unused granule space'
        
        legend_y += 25
        
        # Add bottom padding
        legend_y += 40
        
        return ET.tostring(svg, encoding='unicode')

def write_stats_to_csv(stats, csv_filename):
    if not stats:
        return
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)

def main():
    """Main entry point for the flit packing system"""
    parser = argparse.ArgumentParser(description='Flit Packet Packing System with Header-Meta-Data Support')
    parser.add_argument('--config', '-c', 
                       default='flit_config.json',
                       help='Path to configuration JSON file (default: flit_config.json)')
    parser.add_argument('--output-prefix', '-o',
                       default='',
                       help='Prefix for output SVG files (default: none)')
    parser.add_argument('--generate-all', '-a',
                       action='store_true',
                       help='Generate visualizations for all test distributions')
    parser.add_argument('--max-flits-visualize', '-m',
                       type=int,
                       default=3,
                       help='Maximum number of flits to visualize per test (default: 3)')
    parser.add_argument('--csv-stats', type=str, help='Output CSV file for test run statistics')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_configuration(args.config)
    
    # Extract configuration
    layout_config = config["layout"]
    packet_types = config["packet_types"]
    test_packet_configs = config["test_packets"]
    test_distribution_configs = config["test_distributions"]
    
    # Extract packet sizes
    packet_sizes = []
    for packet_type in packet_types.values():
        total_size = packet_type["header_size"] + packet_type["meta_size"] + packet_type["data_size"]
        if total_size not in packet_sizes:
            packet_sizes.append(total_size)
    packet_sizes.sort()
    
    # Create system components
    layout = FlitLayout(
        flit_size=layout_config["flit_size"],
        header_positions=layout_config["header_positions"],
        granule_size=layout_config["granule_size"],
        num_granules=layout_config["num_granules"]
    )
    
    packer = FlitPacker(layout, packet_sizes)
    
    print("Flit Packing System with Header-Meta-Data Support")
    print("=" * 50)
    print(f"Configuration: {args.config}")
    print(f"Flit size: {layout.flit_size} bytes")
    print(f"Granule size: {layout.granule_size} bytes")
    print(f"Granules per flit: {layout.num_granules}")
    print()
    
    # Display packet types
    print("Packet Types:")
    for type_key, packet_type in packet_types.items():
        total_size = packet_type["header_size"] + packet_type["meta_size"] + packet_type["data_size"]
        print(f"  {type_key}: {packet_type['name']} ({packet_type['header_size']}H + {packet_type['meta_size']}M + {packet_type['data_size']}D = {total_size}B)")
    print()
    
    # Create test packets
    test_packets = []
    for packet_config in test_packet_configs:
        packet = create_packet_from_type(
            packet_config["type"], 
            packet_types, 
            packet_config["packet_id"]
        )
        test_packets.append(packet)
    
    # Pack and analyze example test packets
    print("Analyzing example test packets...")
    flits, granules, stats = packer.pack_packets(test_packets)
    
    print("Example Results:")
    print(f"  Packets: {len(test_packets)}")
    print(f"  Flits: {stats['total_flits']}")
    print(f"  Data Efficiency: {stats['data_efficiency']:.1f}%")
    print(f"  Overall Efficiency: {stats['overall_efficiency']:.1f}%")
    print()
    
    # Create visualizations
    visualizer = FlitVisualizer(layout)
    
    def get_filename(base_name: str) -> str:
        return f"{args.output_prefix}{base_name}" if args.output_prefix else base_name
    
    # Generate structure SVG
    print("Generating visualizations...")
    svg_content = visualizer.create_structure_svg()
    structure_file = get_filename('flit_structure.svg')
    with open(structure_file, 'w') as f:
        f.write(svg_content)
    print(f"✓ Structure: {structure_file}")
    
    # Generate visualization for example test packets
    # Limit the number of flits for visualization to keep files manageable
    flits_to_visualize = flits[:args.max_flits_visualize]
    granules_to_visualize = granules[:args.max_flits_visualize]
    
    packed_svg = visualizer.create_packed_flits_svg(
        flits_to_visualize, 
        granules_to_visualize, 
        test_packets, 
        stats, 
        "Example Test Packets"
    )
    example_file = get_filename('example_packing.svg')
    with open(example_file, 'w') as f:
        f.write(packed_svg)
    print(f"✓ Example packing: {example_file}")
    
    # Run efficiency analysis and generate visualizations for distributions
    analyzer = PackingAnalyzer(packer, packet_types)
    
    print("\nRunning efficiency analysis...")
    for i, dist_config in enumerate(test_distribution_configs, 1):
        distribution = {}
        for packet_type_name, count in dist_config["distribution"].items():
            if packet_type_name in packet_types:
                packet_type = packet_types[packet_type_name]
                total_size = packet_type["header_size"] + packet_type["meta_size"] + packet_type["data_size"]
                distribution[total_size] = count
        
        # Run efficiency analysis
        results = analyzer.analyze_efficiency(distribution, 100)  # Reduced iterations for demo
        print(f"  Test {i} ({dist_config['name']}): {results['overall_efficiency_mean']:.1f}% overall efficiency")
        
        # Generate visualization for this distribution if requested
        if args.generate_all:
            print(f"    Generating visualization for {dist_config['name']}...")
            
            # Create a single random instance of this distribution for visualization
            dist_packets = []
            packet_id = 100 + i * 100  # Start with unique IDs for each distribution
            
            for packet_type_name, count in dist_config["distribution"].items():
                if packet_type_name in packet_types:
                    for j in range(count):
                        packet = create_packet_from_type(packet_type_name, packet_types, packet_id)
                        dist_packets.append(packet)
                        packet_id += 1
            
            # Shuffle for realistic packing scenario
            random.shuffle(dist_packets)
            
            # Pack the packets
            dist_flits, dist_granules, dist_stats = packer.pack_packets(dist_packets)
            
            # Limit visualization to manageable number of flits
            dist_flits_viz = dist_flits[:args.max_flits_visualize]
            dist_granules_viz = dist_granules[:args.max_flits_visualize]
            
            # Generate packed flits visualization
            dist_svg = visualizer.create_packed_flits_svg(
                dist_flits_viz,
                dist_granules_viz,
                dist_packets,
                dist_stats,
                dist_config['name']
            )
            
            # Save with distribution-specific name
            dist_filename = dist_config['name'].lower().replace(' ', '_')
            dist_file = get_filename(f'distribution_{i}_{dist_filename}.svg')
            with open(dist_file, 'w') as f:
                f.write(dist_svg)
            print(f"    ✓ Distribution visualization: {dist_file}")
    
    # Generate comparison visualization showing different packet arrangements
    if args.generate_all:
        print("\nGenerating packet arrangement comparisons...")
        
        # Create a worst-case and best-case scenario with the same packets
        comparison_packets = test_packets[:4]  # Use first 4 packets for comparison
        
        # Worst case: packets in size order (largest first)
        worst_case_packets = sorted(comparison_packets, key=lambda p: p.size, reverse=True)
        worst_flits, worst_granules, worst_stats = packer.pack_packets(worst_case_packets)
        
        worst_svg = visualizer.create_packed_flits_svg(
            worst_flits[:args.max_flits_visualize],
            worst_granules[:args.max_flits_visualize],
            worst_case_packets,
            worst_stats,
            "Worst Case (Largest First)"
        )
        worst_file = get_filename('worst_case_packing.svg')
        with open(worst_file, 'w') as f:
            f.write(worst_svg)
        print(f"✓ Worst case packing: {worst_file}")
        
        # Best case: packets in optimal order (smallest first)
        best_case_packets = sorted(comparison_packets, key=lambda p: p.size)
        best_flits, best_granules, best_stats = packer.pack_packets(best_case_packets)
        
        best_svg = visualizer.create_packed_flits_svg(
            best_flits[:args.max_flits_visualize],
            best_granules[:args.max_flits_visualize],
            best_case_packets,
            best_stats,
            "Best Case (Smallest First)"
        )
        best_file = get_filename('best_case_packing.svg')
        with open(best_file, 'w') as f:
            f.write(best_svg)
        print(f"✓ Best case packing: {best_file}")
        
        print(f"\nEfficiency comparison:")
        print(f"  Worst case (largest first): {worst_stats['overall_efficiency']:.1f}%")
        print(f"  Best case (smallest first): {best_stats['overall_efficiency']:.1f}%")
        print(f"  Difference: {best_stats['overall_efficiency'] - worst_stats['overall_efficiency']:.1f} percentage points")
    
    print(f"\nVisualization Summary:")
    print(f"  Files saved with prefix: '{args.output_prefix}'")
    if not args.generate_all:
        print(f"  Use --generate-all (-a) to create visualizations for all test distributions")
    print(f"  Use --max-flits-visualize (-m) to control how many flits are visualized")
    print("  Run with --help for more options")
    
    # Assume test run statistics are collected in a list of dicts called test_stats
    # test_stats = [...]  # ...existing code...

    if args.csv_stats:
        write_stats_to_csv([worst_stats], args.csv_stats)

if __name__ == "__main__":
    main()