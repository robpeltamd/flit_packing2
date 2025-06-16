from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .packet import Packet

@dataclass
class Granule:
    """Represents a granule within a flit"""
    granule_id: int
    size: int
    flit_id: int
    granule_type: str = "data"  # "header", "meta", or "data"
    packet_id: int = None
    bytes_used: int = 0
    data: bytes = None

    def __post_init__(self):
        if self.data is None:
            self.data = bytearray(self.size)

    def can_fit_packet_data(self, packet_size: int, data_type: str = "data") -> bool:
        return (self.bytes_used + packet_size <= self.size and 
                self.packet_id is None and 
                self.granule_type == data_type)

    def add_packet_data(self, packet: 'Packet', data_size: int, data_type: str = "data") -> int:
        if self.packet_id is not None and self.packet_id != packet.packet_id:
            return 0
        if self.granule_type != data_type:
            return 0
        available_space = self.size - self.bytes_used
        bytes_to_add = min(data_size, available_space)
        if bytes_to_add > 0:
            self.packet_id = packet.packet_id
            if data_type == "header":
                pattern = (packet.packet_id % 256) | 0x80
            elif data_type == "meta":
                pattern = (packet.packet_id % 256) | 0x40
            else:
                pattern = packet.packet_id % 256
            for i in range(bytes_to_add):
                self.data[self.bytes_used + i] = pattern
            self.bytes_used += bytes_to_add
        return bytes_to_add
