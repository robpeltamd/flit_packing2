from dataclasses import dataclass

@dataclass
class Packet:
    """Represents a byte packet with size and ID"""
    size: int
    packet_id: int
    name: str = ""
    header_size: int = 0
    meta_size: int = 0
    data_size: int = 0
    data: bytes = None

    def __post_init__(self):
        if self.data is None:
            self.data = bytes([self.packet_id % 256] * self.size)
        if self.header_size > 0 or self.meta_size > 0 or self.data_size > 0:
            self.size = self.header_size + self.meta_size + self.data_size
