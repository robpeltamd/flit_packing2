import random
import statistics
from dataclasses import dataclass
from typing import List, Tuple, Dict
import xml.etree.ElementTree as ET
import json
import os
import sys

@dataclass
class Packet:
    """Represents a byte packet with size and ID"""
    size: int
    packet_id: int
    data: bytes = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = bytes([self.packet_id % 256] * self.size)

@dataclass
class Granule:
    """Represents a granule within a flit"""
    granule_id: int
    size: int
    flit_id: int
    packet_id: int = None  # Which packet this granule belongs to
    bytes_used: int = 0
    data: bytes = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = bytearray(self.size)
    
    def can_fit_packet_data(self, packet_size: int) -> bool:
        """Check if packet data can fit in this granule"""
        return self.bytes_used + packet_size <= self.size and self.packet_id is None
    
    def add_packet_data(self, packet: Packet, data_size: int) -> int:
        """Add packet data to granule, returns bytes actually added"""
        if self.packet_id is not None and self.packet_id != packet.packet_id:
            return 0  # Cannot mix packets in same granule
        
        available_space = self.size - self.bytes_used
        bytes_to_add = min(data_size, available_space)
        
        if bytes_to_add > 0:
            self.packet_id = packet.packet_id
            # Fill with packet data pattern
            for i in range(bytes_to_add):
                self.data[self.bytes_used + i] = packet.packet_id % 256
            self.bytes_used += bytes_to_add
        
        return bytes_to_add

@dataclass
class FlitLayout:
    """Defines the flit structure with header positions and granules"""
    flit_size: int = 256
    header_positions: List[int] = None
    granule_size: int = 20
    num_granules: int = None
    
    def __post_init__(self):
        if self.header_positions is None:
            # Default header positions: bytes 0, 64, 128, 192
            self.header_positions = [0, 64, 128, 192]
        
        # Calculate data capacity
        self.total_header_bytes = len(self.header_positions)
        self.data_capacity = self.flit_size - self.total_header_bytes
        
        # Set default number of granules if not specified
        if self.num_granules is None:
            self.num_granules = self.data_capacity // self.granule_size
        
        # Validate granule configuration
        self._validate_granule_config()
    
    def _validate_granule_config(self):
        """Validate that granule configuration is valid"""
        required_data_bytes = self.num_granules * self.granule_size
        
        if required_data_bytes > self.data_capacity:
            raise ValueError(
                f"Granule configuration invalid: {self.num_granules} granules × "
                f"{self.granule_size} bytes = {required_data_bytes} bytes, "
                f"but only {self.data_capacity} data bytes available "
                f"({self.flit_size} - {self.total_header_bytes} headers)"
            )
        
        if required_data_bytes != self.data_capacity:
            print(f"Warning: Granules use {required_data_bytes} bytes, "
                  f"leaving {self.data_capacity - required_data_bytes} bytes unused per flit")
    
    def get_available_positions(self) -> List[Tuple[int, int]]:
        """Returns list of (start, end) tuples for available data regions"""
        positions = []
        headers = sorted(self.header_positions)
        
        start = 0
        for header_pos in headers:
            if start < header_pos:
                positions.append((start, header_pos))
            start = header_pos + 1
        
        # Add final segment if there's space after last header
        if start < self.flit_size:
            positions.append((start, self.flit_size))
            
        return positions
    
    def get_granule_layout(self) -> List[Tuple[int, int, int]]:
        """Returns list of (start, end, granule_id) byte positions for each granule"""
        available_regions = self.get_available_positions()
        granule_positions = []
        
        current_granule = 0
        granule_start_byte = 0
        
        for region_start, region_end in available_regions:
            region_pos = region_start
            
            while region_pos < region_end and current_granule < self.num_granules:
                # Calculate how much of current granule fits in this region
                granule_bytes_placed = granule_start_byte
                granule_bytes_remaining = self.granule_size - granule_bytes_placed
                region_bytes_available = region_end - region_pos
                
                bytes_to_place = min(granule_bytes_remaining, region_bytes_available)
                
                granule_positions.append((region_pos, region_pos + bytes_to_place, current_granule))
                
                granule_start_byte += bytes_to_place
                region_pos += bytes_to_place
                
                # Check if granule is complete
                if granule_start_byte >= self.granule_size:
                    current_granule += 1
                    granule_start_byte = 0
        
        return granule_positions

class FlitPacker:
    """Handles packing of packets into granules and flits"""
    
    def __init__(self, layout: FlitLayout, packet_sizes: List[int]):
        self.layout = layout
        self.packet_sizes = packet_sizes
        self.available_regions = layout.get_available_positions()
        self.granule_layout = layout.get_granule_layout()
        self.total_data_capacity = layout.data_capacity
        self.granules_per_flit = layout.num_granules
        self.granule_size = layout.granule_size
    
    def pack_packets(self, packets: List[Packet]) -> Tuple[List[bytes], List[List[Granule]], Dict]:
        """Pack packets into granules and flits, return flits, granules, and statistics"""
        flits = []
        all_granules = []  # List of lists, each containing granules for one flit
        current_flit = bytearray(self.layout.flit_size)
        current_granules = []
        
        # Initialize headers (using 0xFF as header marker)
        for pos in self.layout.header_positions:
            current_flit[pos] = 0xFF
        
        # Create initial set of granules for first flit
        for i in range(self.granules_per_flit):
            granule = Granule(
                granule_id=i,
                size=self.granule_size,
                flit_id=len(flits)
            )
            current_granules.append(granule)
        
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
            remaining_packet_size = packet.size
            packet_fragmented = False
            
            while remaining_packet_size > 0:
                # Find a granule that can fit some of this packet
                granule_found = False
                
                for granule in current_granules:
                    if granule.packet_id is None or granule.packet_id == packet.packet_id:
                        # Check if we can add data to this granule
                        available_in_granule = granule.size - granule.bytes_used
                        if available_in_granule > 0:
                            bytes_added = granule.add_packet_data(packet, remaining_packet_size)
                            if bytes_added > 0:
                                remaining_packet_size -= bytes_added
                                packed_bytes += bytes_added
                                granule_found = True
                                break
                
                if not granule_found:
                    # No suitable granule found, need a new flit
                    
                    # Calculate wasted bytes in current granules
                    for granule in current_granules:
                        wasted_granule_bytes += granule.size - granule.bytes_used
                    
                    # Place granules into current flit
                    self._place_granules_in_flit(current_flit, current_granules)
                    
                    # Save current flit
                    flits.append(bytes(current_flit))
                    all_granules.append(current_granules)
                    
                    # Start new flit
                    current_flit = bytearray(self.layout.flit_size)
                    current_granules = []
                    
                    # Initialize headers
                    for pos in self.layout.header_positions:
                        current_flit[pos] = 0xFF
                    
                    # Create new granules
                    for i in range(self.granules_per_flit):
                        granule = Granule(
                            granule_id=i,
                            size=self.granule_size,
                            flit_id=len(flits)
                        )
                        current_granules.append(granule)
                    
                    if remaining_packet_size > 0:
                        packet_fragmented = True
                
                # Safety check to prevent infinite loop
                if remaining_packet_size > 0 and not granule_found and not current_granules:
                    print(f"Warning: Cannot place packet {packet.packet_id} of size {remaining_packet_size}")
                    break
            
            if packet_fragmented:
                fragmented_packets += 1
        
        # Handle the last flit if it has data
        if any(g.bytes_used > 0 for g in current_granules):
            # Calculate wasted bytes in final granules
            for granule in current_granules:
                wasted_granule_bytes += granule.size - granule.bytes_used
            
            self._place_granules_in_flit(current_flit, current_granules)
            flits.append(bytes(current_flit))
            all_granules.append(current_granules)
        
        # Calculate statistics (excluding the last flit from efficiency calculations)
        stats['total_flits'] = len(flits)
        stats['total_granules'] = len(flits) * self.granules_per_flit
        stats['packed_bytes'] = packed_bytes
        stats['fragmented_packets'] = fragmented_packets
        stats['wasted_granule_bytes'] = wasted_granule_bytes
        
        # For efficiency calculations, exclude the last flit as it's always partially filled
        flits_for_efficiency = max(0, len(flits) - 1) if len(flits) > 1 else 0
        granules_for_efficiency = flits_for_efficiency * self.granules_per_flit
        
        total_capacity_for_efficiency = flits_for_efficiency * self.total_data_capacity
        total_granule_capacity_for_efficiency = granules_for_efficiency * self.granule_size
        
        # Calculate efficiency based on full flits only
        if total_capacity_for_efficiency > 0 and flits_for_efficiency > 0:
            # Calculate packed bytes in full flits only (exclude last flit's contribution)
            if len(flits) > 1:
                # Estimate packed bytes in the last flit
                last_flit_granules = all_granules[-1] if all_granules else []
                last_flit_packed_bytes = sum(g.bytes_used for g in last_flit_granules)
                packed_bytes_full_flits = packed_bytes - last_flit_packed_bytes
                
                stats['efficiency'] = (packed_bytes_full_flits / total_capacity_for_efficiency * 100)
                stats['granule_efficiency'] = (packed_bytes_full_flits / total_granule_capacity_for_efficiency * 100)
            else:
                stats['efficiency'] = 0.0
                stats['granule_efficiency'] = 0.0
        else:
            stats['efficiency'] = 0.0
            stats['granule_efficiency'] = 0.0
        
        # Total waste includes all flits
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
            
            # Find all byte positions for this granule
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
    
    def __init__(self, packer: FlitPacker):
        self.packer = packer
    
    def analyze_efficiency(self, packet_counts: Dict[int, int], num_iterations: int = 1000) -> Dict:
        """Analyze packing efficiency over multiple random packet orderings"""
        efficiencies = []
        granule_efficiencies = []
        flit_counts = []
        fragmentation_rates = []
        
        for _ in range(num_iterations):
            # Generate random packet sequence
            packets = []
            packet_id = 0
            
            for size, count in packet_counts.items():
                for _ in range(count):
                    packets.append(Packet(size, packet_id))
                    packet_id += 1
            
            # Randomize order
            random.shuffle(packets)
            
            # Pack and analyze
            flits, granules, stats = self.packer.pack_packets(packets)
            
            efficiencies.append(stats['efficiency'])
            granule_efficiencies.append(stats['granule_efficiency'])
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
    
    def create_structure_svg(self, width: int = 2000, height: int = 4000) -> str:
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
            height = title_height + len(flits) * (rows_per_flit * cell_height + flit_spacing) + 4000  # Added more padding
        
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
        stats_text.text = f"Flit Efficiency: {stats['efficiency']:.1f}% | Granule Efficiency: {stats['granule_efficiency']:.1f}% | Flits: {stats['total_flits']} ({stats.get('full_flits_analyzed', 0)} analyzed) | Fragmented: {stats['fragmented_packets']} packets"
        
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
            for granule in flit_granules:
                granule_usage[granule.granule_id] = granule.bytes_used
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
                    if status == 'used' and granule_id in granule_packets:
                        packet_id = granule_packets[granule_id]
                        cell_class = f'packet{packet_id % len(self.packet_colors)}'
                        cell_text = str(packet_id)
                    elif status == 'unused':
                        cell_class = 'unused-granule'
                        cell_text = 'U'
                    else:
                        cell_class = 'empty'
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
        
        # Unused granule space legend
        unused_rect = ET.SubElement(svg, 'rect', {
            'x': str(margin + 100),
            'y': str(legend_y - 8),
            'width': '12',
            'height': '10',
            'class': 'unused-granule'
        })
        unused_text = ET.SubElement(svg, 'text', {
            'x': str(margin + 118),
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
            packet_text.text = f'P{packet.packet_id}({packet.size}B)'
            
            col_count += 1
        
        # Add bottom padding
        legend_y += 40
        
        return ET.tostring(svg, encoding='unicode')
def load_configuration(config_file: str = "flit_config.json") -> Dict:
    """Load configuration from JSON file, create default if not exists"""
    default_config = {
        "layout": {
            "flit_size": 256,
            "header_positions": [0, 64, 128, 192],
            "granule_size": 20,
            "num_granules": None
        },
        "packet_sizes": [16, 32, 64, 128],
        "test_packets": [
            {"size": 32, "packet_id": 1},
            {"size": 64, "packet_id": 2},
            {"size": 16, "packet_id": 3},
            {"size": 128, "packet_id": 4},
            {"size": 32, "packet_id": 5},
            {"size": 16, "packet_id": 6}
        ],
        "test_distributions": [
            {
                "distribution": {"16": 10, "32": 5, "64": 3, "128": 1},
                "name": "Many Small Packets"
            },
            {
                "distribution": {"16": 2, "32": 2, "64": 2, "128": 4},
                "name": "Many Large Packets"
            },
            {
                "distribution": {"16": 5, "32": 5, "64": 5, "128": 5},
                "name": "Balanced Distribution"
            }
        ]
    }
    
    if not os.path.exists(config_file):
        # Create default config file
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=4)
        print(f"Created default configuration file: {config_file}")
        print("You can edit this file to customize layout, packets, and test distributions.")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from: {config_file}")
        return config
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        sys.exit(1) 

# Example usage and demonstration
def main():
    # Load configuration from file
    config = load_configuration("ualink_over_c2c.json")
    
    # Extract configuration
    layout_config = config["layout"]
    packet_sizes = config["packet_sizes"]
    test_packet_configs = config["test_packets"]
    test_distribution_configs = config["test_distributions"]
    
    # Create flit layout from config
    layout = FlitLayout(
        flit_size=layout_config["flit_size"],
        header_positions=layout_config["header_positions"],
        granule_size=layout_config["granule_size"],
        num_granules=layout_config["num_granules"]
    )
    
    # Create packer
    packer = FlitPacker(layout, packet_sizes)
    
    print("Flit Packing System with Granules")
    print("=" * 50)
    print(f"Configuration loaded from: flit_config.json")
    print(f"Flit size: {layout.flit_size} bytes")
    print(f"Header positions: {layout.header_positions}")
    print(f"Granule size: {layout.granule_size} bytes")
    print(f"Granules per flit: {layout.num_granules}")
    print(f"Available data regions: {layout.get_available_positions()}")
    print(f"Total data capacity per flit: {packer.total_data_capacity} bytes")
    print(f"Data efficiency: {packer.total_data_capacity/layout.flit_size*100:.1f}%")
    print(f"Granule capacity per flit: {layout.num_granules * layout.granule_size} bytes")
    print()
    
    # Create test packets from config
    test_packets = []
    for packet_config in test_packet_configs:
        test_packets.append(Packet(packet_config["size"], packet_config["packet_id"]))
    
    flits, granules, stats = packer.pack_packets(test_packets)
    
    print("Example Packing Results:")
    print(f"Total packets: {len(test_packets)}")
    print(f"Total packet bytes: {stats['total_packet_bytes']}")
    print(f"Flits generated: {stats['total_flits']}")
    print(f"Full flits analyzed for efficiency: {stats['full_flits_analyzed']}")
    print(f"Granules generated: {stats['total_granules']}")
    print(f"Flit efficiency (full flits only): {stats['efficiency']:.2f}%")
    print(f"Granule efficiency (full flits only): {stats['granule_efficiency']:.2f}%")
    print(f"Fragmented packets: {stats['fragmented_packets']}")
    print(f"Total waste bytes: {stats['waste_bytes']}")
    print(f"Wasted granule bytes: {stats['wasted_granule_bytes']}")
    print()
    
    # Create visualizer
    visualizer = FlitVisualizer(layout)
    
    # Generate basic flit structure SVG
    svg_content = visualizer.create_structure_svg()
    try:
        with open('flit_structure.svg', 'w') as f:
            f.write(svg_content)
        print("✓ Basic flit structure saved as 'flit_structure.svg'")
    except Exception as e:
        print(f"✗ Error saving flit_structure.svg: {e}")
    
    # Generate packed example visualization
    packed_svg = visualizer.create_packed_flits_svg(flits, granules, test_packets, stats, "Example Packing")
    try:
        with open('example_packing.svg', 'w') as f:
            f.write(packed_svg)
        print("✓ Example packing visualization saved as 'example_packing.svg'")
    except Exception as e:
        print(f"✗ Error saving example_packing.svg: {e}")
    
    # Efficiency analysis with visualizations
    analyzer = PackingAnalyzer(packer)
    
    # Convert test distributions from config
    test_distributions = []
    for dist_config in test_distribution_configs:
        # Convert string keys to integers
        distribution = {int(k): v for k, v in dist_config["distribution"].items()}
        test_distributions.append((distribution, dist_config["name"]))
    
    print("\nEfficiency Analysis (1000 randomizations each):")
    print("-" * 80)
    
    for i, (distribution, test_name) in enumerate(test_distributions, 1):
        total_packets = sum(distribution.values())
        total_bytes = sum(size * count for size, count in distribution.items())
        
        print(f"\nTest {i}: {test_name}")
        print(f"Total packets: {total_packets}, Total bytes: {total_bytes}")
        
        results = analyzer.analyze_efficiency(distribution, 1000)
        
        print(f"Efficiency: {results['efficiency_mean']:.2f}% ± {results['efficiency_std']:.2f}%")
        print(f"Range: {results['efficiency_min']:.2f}% - {results['efficiency_max']:.2f}%")
        print(f"Granule Efficiency: {results['granule_efficiency_mean']:.2f}% ± {results['granule_efficiency_std']:.2f}%")
        print(f"Avg flits: {results['flit_count_mean']:.2f} ± {results['flit_count_std']:.2f}")
        print(f"Fragmentation rate: {results['fragmentation_rate_mean']:.2f}% ± {results['fragmentation_rate_std']:.2f}%")
        
        # Generate one example packing for this distribution
        packets = []
        packet_id = 0
        
        for size, count in distribution.items():
            for _ in range(count):
                packets.append(Packet(size, packet_id))
                packet_id += 1
        
        # DON'T shuffle - pack in generation order for visualization
        
        # Pack and visualize
        test_flits, test_granules, test_stats = packer.pack_packets(packets)
        
        # Create visualization for this test
        test_svg = visualizer.create_packed_flits_svg(
            test_flits, test_granules, packets, test_stats, f"Test {i}: {test_name}"
        )
        
        filename = f'test_{i}_packing.svg'
        try:
            with open(filename, 'w') as f:
                f.write(test_svg)
            print(f"✓ Visualization saved as '{filename}'")
        except Exception as e:
            print(f"✗ Error saving {filename}: {e}")
    
    print(f"\n" + "="*60)
    print("SVG FILES GENERATED:")
    print("="*60)
    print("✓ flit_structure.svg: Basic flit layout showing headers and granule regions")
    print("✓ example_packing.svg: Example packet packing demonstration")
    print("✓ test_1_packing.svg: Many small packets test visualization")
    print("✓ test_2_packing.svg: Many large packets test visualization") 
    print("✓ test_3_packing.svg: Balanced distribution test visualization")
    print("="*60)
    print("All files saved in the current working directory.")
    print("Open any SVG file in a web browser to view the visualization.")
    print(f"\nTo customize configuration, edit 'flit_config.json' and re-run the script.")

if __name__ == "__main__":
    main()