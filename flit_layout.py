from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class FlitLayout:
    """Defines the flit structure with header positions and granules"""
    flit_size: int = 256
    header_positions: List[int] = None
    granule_size: int = 20
    num_granules: int = None

    def __post_init__(self):
        if self.header_positions is None:
            self.header_positions = [0, 64, 128, 192]
        self.total_header_bytes = len(self.header_positions)
        self.data_capacity = self.flit_size - self.total_header_bytes
        if self.num_granules is None:
            self.num_granules = self.data_capacity // self.granule_size
        self._validate_granule_config()

    def _validate_granule_config(self):
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
        positions = []
        headers = sorted(self.header_positions)
        start = 0
        for header_pos in headers:
            if start < header_pos:
                positions.append((start, header_pos))
            start = header_pos + 1
        if start < self.flit_size:
            positions.append((start, self.flit_size))
        return positions

    def get_granule_layout(self) -> List[Tuple[int, int, int]]:
        available_regions = self.get_available_positions()
        granule_positions = []
        current_granule = 0
        granule_start_byte = 0
        for region_start, region_end in available_regions:
            region_pos = region_start
            while region_pos < region_end and current_granule < self.num_granules:
                granule_bytes_placed = granule_start_byte
                granule_bytes_remaining = self.granule_size - granule_bytes_placed
                region_bytes_available = region_end - region_pos
                bytes_to_place = min(granule_bytes_remaining, region_bytes_available)
                granule_positions.append((region_pos, region_pos + bytes_to_place, current_granule))
                granule_start_byte += bytes_to_place
                region_pos += bytes_to_place
                if granule_start_byte >= self.granule_size:
                    current_granule += 1
                    granule_start_byte = 0
        return granule_positions
