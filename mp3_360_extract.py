#!/usr/bin/env python3
"""
mp3_360_extract.py - Max Payne 3 (Xbox 360) model extractor, single file.

Extracts models and textures from the game's RSC7 resources:
    .xdd (+ .xtd)  pedestrian / drawable dictionary  -> OBJ + MTL + PNG
    .xdr (+ .xtd)  single drawable                    -> OBJ + MTL + PNG
    .xft (+ .xtd)  vehicle fragment (body + wheels)   -> OBJ + MTL + PNG
    .xtd           texture dictionary                 -> PNG

Usage:
    python3 mp3_360_extract.py <folder>            -o out_dir   convert everything in a folder
    python3 mp3_360_extract.py model.xdd [tex.xtd] -o out_dir   single ped / drawable dictionary
    python3 mp3_360_extract.py model.xdr [tex.xtd] -o out_dir   single drawable
    python3 mp3_360_extract.py car.xft   [car.xtd] -o out_dir   vehicle
    python3 mp3_360_extract.py textures.xtd        -o out_dir   textures only (PNG)

Folder mode walks sub-folders (use --no-recursive to stop that), mirrors the folder
structure in the output, keeps going after errors and prints a summary.

Where a model's textures come from (models and .xtd files do NOT have to share a name).
A model can combine several sources; each one only supplies what the previous lacked:
  1. the .xtd with the same name as the model, if there is one;
  2. the texture dictionary embedded in the model itself (many props carry their own);
  3. .xtd files in --tex-dir folders;
  4. shared texture archives: .xtd files that belong to no model (no .xdd/.xdr/.xft with
     the same name), found anywhere under the input folder. The archive containing the
     most of the still-missing textures wins (ties: closest file name, same folder first).
A .xtd that belongs to another model (e.g. another ped's) is never used in step 4:
peds reuse texture names with different content, so borrowing would be wrong.
The sources used are printed for every model. Textures from a shared archive are
written once to <thatxtd>_textures/ and every model's .mtl points there; own and
embedded textures go to <model>_textures/.
A .xtd without a model is exported as PNGs. A .xft next to a same-named .xdd is that
ped's fragment and is skipped.

Supported: .xdd .xdr .xft .xtd. Recognised but NOT converted (no samples to verify
against): .xbn/.xbd collision bounds, .xpl placements - they are listed as skipped.
Texture formats: DXT1, DXT3, DXT5 and DXT5A (single channel, shown as grey).
Two-channel normal maps (blue = 0 in the file) get Z rebuilt so they work in viewers.
Diffuse textures with real transparency (glass, hair, decals) get "d 1.0" + "map_d" in the
.mtl, pointing at the same RGBA PNG.
Container variants: RSC7 (16-byte header), RSC5 (20-byte header with the 0x0FF512F1
compression marker) and re-wrapped RSC7 files (22-byte prefix) are all read.
Requires: Python 3.9+, numpy, Pillow.

How it works (all verified against real files):
  1. RSC7 header (16 bytes: magic, version, flags, pad). Flags give the
     virtual/physical segment sizes. Payload = LZXD stream (128KB window)
     in Xbox 360 chunk framing: 2-byte big-endian compressed size per 32KB
     chunk, or 0xFF + 2-byte uncompressed size + 2-byte compressed size.
     The LZXD decoder is a pure-Python port of the Rust `lzxd` crate.
  2. Decoded buffer = [virtual][physical]; pointers 0x5xxxxxxx / 0x6xxxxxxx,
     everything big-endian.
  3. Geometry: drawable -> LOD -> models -> geometries -> vertex/index
     buffers, vertex layout from the vertex declaration.
  4. Textures: the fetch descriptor is at (header+0x1C pointer)+0x20; mip0
     page, GPU format (0x12 BC1, 0x14 BC3), size; data is 16-bit
     byte-swapped and Xbox 360 tiled (32x32 block padded grid).
  5. Vehicles: the fragment holds the body drawable (0xB4) and children
     (0xD4); one wheel model is instanced at bones 33..36.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import struct
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


# ============================================================================
# LZXD decompressor (port of the Rust `lzxd` crate)
# ============================================================================

class DecodeFailed(Exception):
    pass


MAX_CHUNK_SIZE = 32 * 1024

# window sizes -> (value, position_slots)
_WINDOW_SIZES = {
    "KB32": (0x8000, 30),
    "KB64": (0x10000, 32),
    "KB128": (0x20000, 34),
    "KB256": (0x40000, 36),
    "KB512": (0x80000, 38),
    "MB1": (0x100000, 42),
    "MB2": (0x200000, 50),
    "MB4": (0x400000, 66),
    "MB8": (0x800000, 98),
    "MB16": (0x1000000, 162),
    "MB32": (0x2000000, 290),
}


def window_size_for(nbytes: int) -> str:
    """Pick the smallest supported window size >= nbytes (and >= MAX_CHUNK_SIZE)."""
    for name, (val, _) in _WINDOW_SIZES.items():
        if val >= nbytes and val >= MAX_CHUNK_SIZE:
            return name
    return "MB32"


# ---------------------------------------------------------------------------
# Bitstream
# ---------------------------------------------------------------------------

class Bitstream:
    """LZXD bitstream: sequence of little-endian 16-bit words, MSB-first bit order."""

    __slots__ = ("buffer", "pos", "n", "remaining")

    def __init__(self, buffer: bytes):
        self.buffer = buffer
        self.pos = 0  # index into buffer of next unread byte
        self.n = 0
        self.remaining = 0

    def _advance_buffer(self):
        if len(self.buffer) - self.pos < 2:
            raise DecodeFailed("UnexpectedEof")
        self.remaining = 16
        self.n = self.buffer[self.pos] | (self.buffer[self.pos + 1] << 8)
        self.pos += 2

    @staticmethod
    def _rol16(v: int, n: int) -> int:
        n &= 15
        if n == 0:
            return v & 0xFFFF
        return ((v << n) | (v >> (16 - n))) & 0xFFFF

    def read_bit(self) -> int:
        if self.remaining == 0:
            self._advance_buffer()
        self.remaining -= 1
        self.n = self._rol16(self.n, 1)
        return self.n & 1

    def read_byte(self) -> Optional[int]:
        if self.pos >= len(self.buffer):
            return None
        b = self.buffer[self.pos]
        self.pos += 1
        return b

    def _read_bits_oneword(self, bits: int) -> int:
        assert bits <= 16
        if bits <= self.remaining:
            self.remaining -= bits
            self.n = self._rol16(self.n, bits)
            return self.n & ((1 << bits) - 1)
        else:
            hi = self._rol16(self.n, self.remaining) & ((1 << self.remaining) - 1)
            bits2 = bits - self.remaining
            self._advance_buffer()
            self.remaining -= bits2
            self.n = self._rol16(self.n, bits2)
            lo_mask = ((1 << bits2) & 0xFFFF) - 1 if bits2 < 16 else 0xFFFF
            lo = self.n & lo_mask
            return ((hi << bits2) | lo) & 0xFFFFFFFF

    def read_bits(self, bits: int) -> int:
        if bits == 0:
            return 0
        if bits <= 16:
            return self._read_bits_oneword(bits)
        assert bits <= 32
        w0 = self._read_bits_oneword(16)
        w1 = self._read_bits_oneword(bits - 16)
        return ((w0 << (bits - 16)) | w1) & 0xFFFFFFFF

    def _peek_bits_oneword(self, bits: int) -> int:
        assert bits <= 16
        if bits <= self.remaining:
            return self._rol16(self.n, bits) & ((1 << bits) - 1)
        else:
            hi = self._rol16(self.n, self.remaining) & ((1 << self.remaining) - 1)
            bits2 = bits - self.remaining
            if len(self.buffer) - self.pos < 2:
                nn = 0
            else:
                nn = self.buffer[self.pos] | (self.buffer[self.pos + 1] << 8)
            lo_mask = ((1 << bits2) & 0xFFFF) - 1 if bits2 < 16 else 0xFFFF
            lo = self._rol16(nn, bits2) & lo_mask
            return ((hi << bits2) | lo) & 0xFFFFFFFF

    def peek_bits(self, bits: int) -> int:
        if bits <= 16:
            return self._peek_bits_oneword(bits)
        assert bits <= 32
        saved = (self.buffer, self.pos, self.n, self.remaining)
        w0 = self._read_bits_oneword(16)
        w1 = self._peek_bits_oneword(bits - 16)
        self.buffer, self.pos, self.n, self.remaining = saved
        return ((w0 << (bits - 16)) | w1) & 0xFFFFFFFF

    def read_u32_le(self) -> int:
        lo = self._read_bits_oneword(16)
        hi = self._read_bits_oneword(16)
        lo_b = lo.to_bytes(2, "little")
        hi_b = hi.to_bytes(2, "little")
        return int.from_bytes(lo_b + hi_b, "little")

    def read_u24_be(self) -> int:
        hi = self.read_bits(16)
        lo = self.read_bits(8)
        return (hi << 8) | lo

    def align(self):
        if self.remaining == 0:
            self.read_bits(16)
        else:
            self.remaining = 0

    def read_raw(self, n: int) -> bytes:
        if len(self.buffer) - self.pos < n:
            raise DecodeFailed("UnexpectedEof")
        out = self.buffer[self.pos:self.pos + n]
        self.pos += n
        return out

    def remaining_bytes(self) -> int:
        return len(self.buffer) - self.pos


# ---------------------------------------------------------------------------
# Huffman trees
# ---------------------------------------------------------------------------

class Tree:
    __slots__ = ("path_lengths", "largest_length", "huffman_tree")

    def __init__(self, path_lengths: List[int]):
        largest_length = max(path_lengths) if path_lengths else 0
        if largest_length == 0:
            raise DecodeFailed("EmptyTree")
        self.path_lengths = path_lengths
        self.largest_length = largest_length
        table_size = 1 << largest_length
        huffman_tree = [0] * table_size

        pos = 0
        for bit in range(1, largest_length + 1):
            amount = 1 << (largest_length - bit)
            for code, pl in enumerate(path_lengths):
                if pl == bit:
                    if pos + amount > table_size:
                        raise DecodeFailed("InvalidPathLengths")
                    for k in range(pos, pos + amount):
                        huffman_tree[k] = code
                    pos += amount
        if pos != table_size:
            raise DecodeFailed("InvalidPathLengths")
        self.huffman_tree = huffman_tree

    def decode_element(self, bitstream: Bitstream) -> int:
        code = self.huffman_tree[bitstream.peek_bits(self.largest_length)]
        bitstream.read_bits(self.path_lengths[code])
        return code


class CanonicalTree:
    __slots__ = ("path_lengths",)

    def __init__(self, count: int):
        self.path_lengths = [0] * count

    def create_instance_allow_empty(self) -> Optional[Tree]:
        if max(self.path_lengths, default=0) == 0:
            return None
        return Tree(list(self.path_lengths))

    def create_instance(self) -> Tree:
        t = self.create_instance_allow_empty()
        if t is None:
            raise DecodeFailed("EmptyTree")
        return t

    def update_range_with_pretree(self, bitstream: Bitstream, start: int, end: int):
        pretree_lengths = [bitstream.read_bits(4) for _ in range(20)]
        pretree = Tree(pretree_lengths)

        i = start
        while i < end:
            code = pretree.decode_element(bitstream)
            if 0 <= code <= 16:
                self.path_lengths[i] = (17 + self.path_lengths[i] - code) % 17
                i += 1
            elif code == 17:
                zeros = bitstream.read_bits(4)
                n = zeros + 4
                if i + n > len(self.path_lengths):
                    raise DecodeFailed("InvalidPretreeRle")
                for k in range(i, i + n):
                    self.path_lengths[k] = 0
                i += n
            elif code == 18:
                zeros = bitstream.read_bits(5)
                n = zeros + 20
                if i + n > len(self.path_lengths):
                    raise DecodeFailed("InvalidPretreeRle")
                for k in range(i, i + n):
                    self.path_lengths[k] = 0
                i += n
            elif code == 19:
                same = bitstream.read_bits(1)
                code2 = pretree.decode_element(bitstream)
                if code2 > 16:
                    raise DecodeFailed("InvalidPretreeElement")
                value = (17 + self.path_lengths[i] - code2) % 17
                n = same + 4
                if i + n > len(self.path_lengths):
                    raise DecodeFailed("InvalidPretreeRle")
                for k in range(i, i + n):
                    self.path_lengths[k] = value
                i += n
            else:
                raise DecodeFailed("InvalidPretreeElement")


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

FOOTER_BITS = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13, 14, 14, 15, 15, 16, 16,
] + [17] * (289 - 36)

BASE_POSITION = [
    0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536,
    2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072, 196608,
    262144, 393216, 524288, 655360, 786432, 917504, 1048576, 1179648, 1310720, 1441792, 1572864,
    1703936, 1835008, 1966080, 2097152, 2228224, 2359296, 2490368, 2621440, 2752512, 2883584,
    3014656, 3145728, 3276800, 3407872, 3538944, 3670016, 3801088, 3932160, 4063232, 4194304,
    4325376, 4456448, 4587520, 4718592, 4849664, 4980736, 5111808, 5242880, 5373952, 5505024,
    5636096, 5767168, 5898240, 6029312, 6160384, 6291456, 6422528, 6553600, 6684672, 6815744,
    6946816, 7077888, 7208960, 7340032, 7471104, 7602176, 7733248, 7864320, 7995392, 8126464,
    8257536, 8388608, 8519680, 8650752, 8781824, 8912896, 9043968, 9175040, 9306112, 9437184,
    9568256, 9699328, 9830400, 9961472, 10092544, 10223616, 10354688, 10485760, 10616832, 10747904,
    10878976, 11010048, 11141120, 11272192, 11403264, 11534336, 11665408, 11796480, 11927552,
    12058624, 12189696, 12320768, 12451840, 12582912, 12713984, 12845056, 12976128, 13107200,
    13238272, 13369344, 13500416, 13631488, 13762560, 13893632, 14024704, 14155776, 14286848,
    14417920, 14548992, 14680064, 14811136, 14942208, 15073280, 15204352, 15335424, 15466496,
    15597568, 15728640, 15859712, 15990784, 16121856, 16252928, 16384000, 16515072, 16646144,
    16777216, 16908288, 17039360, 17170432, 17301504, 17432576, 17563648, 17694720, 17825792,
    17956864, 18087936, 18219008, 18350080, 18481152, 18612224, 18743296, 18874368, 19005440,
    19136512, 19267584, 19398656, 19529728, 19660800, 19791872, 19922944, 20054016, 20185088,
    20316160, 20447232, 20578304, 20709376, 20840448, 20971520, 21102592, 21233664, 21364736,
    21495808, 21626880, 21757952, 21889024, 22020096, 22151168, 22282240, 22413312, 22544384,
    22675456, 22806528, 22937600, 23068672, 23199744, 23330816, 23461888, 23592960, 23724032,
    23855104, 23986176, 24117248, 24248320, 24379392, 24510464, 24641536, 24772608, 24903680,
    25034752, 25165824, 25296896, 25427968, 25559040, 25690112, 25821184, 25952256, 26083328,
    26214400, 26345472, 26476544, 26607616, 26738688, 26869760, 27000832, 27131904, 27262976,
    27394048, 27525120, 27656192, 27787264, 27918336, 28049408, 28180480, 28311552, 28442624,
    28573696, 28704768, 28835840, 28966912, 29097984, 29229056, 29360128, 29491200, 29622272,
    29753344, 29884416, 30015488, 30146560, 30277632, 30408704, 30539776, 30670848, 30801920,
    30932992, 31064064, 31195136, 31326208, 31457280, 31588352, 31719424, 31850496, 31981568,
    32112640, 32243712, 32374784, 32505856, 32636928, 32768000, 32899072, 33030144, 33161216,
    33292288, 33423360,
]


@dataclass
class DecoderState:
    window_size_name: str
    position_slots: int
    main_tree: CanonicalTree = field(init=False)
    length_tree: CanonicalTree = field(init=False)

    def __post_init__(self):
        self.main_tree = CanonicalTree(256 + 8 * self.position_slots)
        self.length_tree = CanonicalTree(249)


# Decoded element kinds
DEC_SINGLE = 0
DEC_MATCH = 1
DEC_READ = 2

KIND_VERBATIM = 0
KIND_ALIGNED = 1
KIND_UNCOMPRESSED = 2


class Block:
    __slots__ = ("remaining", "size", "kind", "main_tree", "length_tree",
                 "aligned_offset_tree", "r_uncompressed")

    def __init__(self):
        self.remaining = 0
        self.size = 0
        self.kind = KIND_UNCOMPRESSED
        self.main_tree: Optional[Tree] = None
        self.length_tree: Optional[Tree] = None
        self.aligned_offset_tree: Optional[Tree] = None
        self.r_uncompressed: Optional[Tuple[int, int, int]] = None

    @staticmethod
    def read(bitstream: Bitstream, state: DecoderState) -> "Block":
        kind_bits = bitstream.read_bits(3)
        size = bitstream.read_u24_be()
        if size == 0:
            raise DecodeFailed("InvalidBlockSize")

        blk = Block()
        blk.remaining = size
        blk.size = size

        if kind_bits == 0b001:
            _read_main_and_length_trees(bitstream, state)
            blk.kind = KIND_VERBATIM
            blk.main_tree = state.main_tree.create_instance()
            blk.length_tree = state.length_tree.create_instance_allow_empty()
        elif kind_bits == 0b010:
            path_lengths = [bitstream.read_bits(3) for _ in range(8)]
            blk.aligned_offset_tree = Tree(path_lengths)
            _read_main_and_length_trees(bitstream, state)
            blk.kind = KIND_ALIGNED
            blk.main_tree = state.main_tree.create_instance()
            blk.length_tree = state.length_tree.create_instance_allow_empty()
        elif kind_bits == 0b011:
            bitstream.align()
            blk.kind = KIND_UNCOMPRESSED
            blk.r_uncompressed = (
                bitstream.read_u32_le(),
                bitstream.read_u32_le(),
                bitstream.read_u32_le(),
            )
        else:
            raise DecodeFailed(f"InvalidBlock({kind_bits})")

        return blk

    def decode_element(self, bitstream: Bitstream, r: List[int]):
        if self.kind == KIND_UNCOMPRESSED:
            r[0], r[1], r[2] = self.r_uncompressed
            return (DEC_READ, self.remaining, None)
        return _decode_element(bitstream, r, self.aligned_offset_tree,
                                self.main_tree, self.length_tree)


def _read_main_and_length_trees(bitstream: Bitstream, state: DecoderState):
    state.main_tree.update_range_with_pretree(bitstream, 0, 256)
    state.main_tree.update_range_with_pretree(
        bitstream, 256, 256 + 8 * state.position_slots)
    state.length_tree.update_range_with_pretree(bitstream, 0, 249)


def _decode_element(bitstream: Bitstream, r: List[int],
                     aligned_offset_tree: Optional[Tree],
                     main_tree: Tree, length_tree: Optional[Tree]):
    main_element = main_tree.decode_element(bitstream)

    if main_element < 256:
        return (DEC_SINGLE, main_element, None)

    length_header = (main_element - 256) & 7
    if length_header == 7:
        if length_tree is None:
            raise DecodeFailed("EmptyTree")
        match_length = length_tree.decode_element(bitstream) + 7 + 2
    else:
        match_length = length_header + 2

    position_slot = (main_element - 256) >> 3

    if position_slot == 0:
        match_offset = r[0]
    elif position_slot == 1:
        match_offset = r[1]
        r[0], r[1] = r[1], r[0]
    elif position_slot == 2:
        match_offset = r[2]
        r[0], r[2] = r[2], r[0]
    else:
        offset_bits = FOOTER_BITS[position_slot]
        if aligned_offset_tree is not None:
            if offset_bits >= 3:
                verbatim_bits = bitstream.read_bits(offset_bits - 3) << 3
                aligned_bits = aligned_offset_tree.decode_element(bitstream)
            else:
                verbatim_bits = bitstream.read_bits(offset_bits)
                aligned_bits = 0
            formatted_offset = BASE_POSITION[position_slot] + verbatim_bits + aligned_bits
        else:
            verbatim_bits = bitstream.read_bits(offset_bits)
            formatted_offset = BASE_POSITION[position_slot] + verbatim_bits

        match_offset = formatted_offset - 2
        r[2] = r[1]
        r[1] = r[0]
        r[0] = match_offset

    return (DEC_MATCH, match_offset, match_length)


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------

class Window:
    __slots__ = ("pos", "buffer", "size")

    def __init__(self, size: int):
        self.pos = 0
        self.size = size
        self.buffer = bytearray(size)

    def _advance(self, delta: int):
        self.pos += delta
        if self.pos >= self.size:
            self.pos -= self.size

    def push(self, value: int):
        self.buffer[self.pos] = value
        self._advance(1)

    def copy_from_self(self, offset: int, length: int):
        if offset <= self.pos and length <= offset and self.pos + length < self.size:
            start = self.pos - offset
            # copy_within semantics: read then write (may overlap forward)
            src = bytes(self.buffer[start:start + length])
            self.buffer[self.pos:self.pos + length] = src
        else:
            mask = self.size - 1
            buf = self.buffer
            pos = self.pos
            for i in range(length):
                dst = (pos + i) & mask
                src = (self.size + pos + i - offset) & mask
                buf[dst] = buf[src]
        self._advance(length)

    def copy_from_bitstream(self, bitstream: Bitstream, length: int):
        if length > self.size:
            raise DecodeFailed("WindowTooSmall")
        if self.pos + length > self.size:
            shift = self.pos + length - self.size
            self.pos -= shift
            self.buffer[0:self.size - shift] = self.buffer[shift:self.size]
        raw = bitstream.read_raw(length)
        self.buffer[self.pos:self.pos + length] = raw
        self._advance(length)

    def past_view(self, length: int) -> bytes:
        if length > MAX_CHUNK_SIZE:
            raise DecodeFailed("ChunkTooLong")
        if self.pos != 0 and length > self.pos:
            shift = length - self.pos
            self._advance(shift)
            tmp = bytes(self.buffer[self.size - shift:self.size])
            self.buffer[shift:self.size] = self.buffer[0:self.size - shift]
            self.buffer[0:shift] = tmp
        pos = self.size if self.pos == 0 else self.pos
        return bytes(self.buffer[pos - length:pos])


# ---------------------------------------------------------------------------
# Top-level decoder
# ---------------------------------------------------------------------------

class Lzxd:
    def __init__(self, window_size_name: str = "KB64", e8_translation: Optional[bool] = None,
                 enable_e8: bool = True):
        val, slots = _WINDOW_SIZES[window_size_name]
        self.window = Window(val)
        self.state = DecoderState(window_size_name, slots)
        self.r = [1, 1, 1]
        self.chunk_offset = 0
        self.first_chunk_read = False
        self.current_block = Block()  # dummy, remaining=0
        self.postprocess_size: Optional[int] = None
        self.postprocess_buf: Optional[bytearray] = None
        self.enable_e8 = enable_e8

    def _try_read_first_chunk(self, bitstream: Bitstream):
        if not self.first_chunk_read:
            self.first_chunk_read = True
            e8 = bitstream.read_bit() != 0
            if e8 and self.enable_e8:
                self.postprocess_size = bitstream.read_bits(32)
                # convert to signed 32-bit
                if self.postprocess_size >= 0x80000000:
                    self.postprocess_size -= 0x100000000
                self.postprocess_buf = bytearray(MAX_CHUNK_SIZE)
            elif e8:
                # e8 flag set but caller disabled translation; still consume the field
                bitstream.read_bits(32)
                self.postprocess_size = None
            else:
                self.postprocess_size = None

    @staticmethod
    def _postprocess(translation_size: int, chunk_offset: int, data: bytearray) -> bytearray:
        processed = 0
        n = len(data)
        while True:
            idx = data.find(0xE8, processed) if False else None
            # bytearray has no .find for int directly in old pythons; use bytes.find
            idx = data.find(b"\xe8", processed)
            if idx == -1:
                break
            if n - idx <= 10:
                break
            current_pointer = chunk_offset + idx
            abs_val = int.from_bytes(data[idx + 1:idx + 5], "little", signed=True)
            if (abs_val >= -current_pointer) and abs_val < translation_size:
                if abs_val >= 0:
                    rel_val = (abs_val - current_pointer) & 0xFFFFFFFF
                else:
                    rel_val = (abs_val + translation_size) & 0xFFFFFFFF
                data[idx + 1:idx + 5] = rel_val.to_bytes(4, "little")
            processed = idx + 5
        return data

    def decompress_next(self, chunk: bytes, output_len: int) -> bytes:
        bitstream = Bitstream(chunk)
        self._try_read_first_chunk(bitstream)

        decoded_len = 0
        while decoded_len != output_len:
            if self.current_block.remaining == 0:
                if self.current_block.kind == KIND_UNCOMPRESSED and self.current_block.size % 2 != 0:
                    bitstream.read_byte()
                self.current_block = Block.read(bitstream, self.state)
                if self.current_block.remaining == 0:
                    raise DecodeFailed("block with zero remaining")

            kind, a, b = self.current_block.decode_element(bitstream, self.r)

            if kind == DEC_SINGLE:
                self.window.push(a)
                advance = 1
            elif kind == DEC_MATCH:
                self.window.copy_from_self(a, b)
                advance = b
            else:  # DEC_READ
                length = min(bitstream.remaining_bytes(), a)
                self.window.copy_from_bitstream(bitstream, length)
                advance = length

            if advance == 0:
                raise DecodeFailed("UnexpectedEof (advance=0)")

            decoded_len += advance
            if self.current_block.remaining < advance:
                raise DecodeFailed("OverreadBlock")
            self.current_block.remaining -= advance

        chunk_offset = self.chunk_offset
        self.chunk_offset += decoded_len

        view = self.window.past_view(decoded_len)
        if self.postprocess_size is not None:
            if chunk_offset >= 0x40000000 or decoded_len <= 10:
                return view
            buf = bytearray(view)
            buf = self._postprocess(self.postprocess_size, chunk_offset, buf)
            return bytes(buf)
        return view



# ============================================================================
# RSC7 container
# ============================================================================

def rsc7_sizes(flags):
    vsize = (flags & 0x7FF) << (((flags >> 11) & 0xF) + 8)
    psize = ((flags >> 15) & 0x7FF) << (((flags >> 26) & 0xF) + 8)
    return vsize, psize


def decode_xmem_lzxd(payload: bytes, target: int, window_size: str = "KB128") -> bytes:
    pos = 0
    out = bytearray()
    lz = Lzxd(window_size)
    while len(out) < target:
        if pos >= len(payload):
            raise DecodeFailed(f"ran out of input at {pos}, produced {len(out)}/{target}")
        if payload[pos] == 0xFF:
            if pos + 5 > len(payload):
                raise DecodeFailed("truncated 0xFF chunk header")
            usize = int.from_bytes(payload[pos + 1:pos + 3], "big")
            csize = int.from_bytes(payload[pos + 3:pos + 5], "big")
            pos += 5
        else:
            if pos + 2 > len(payload):
                raise DecodeFailed("truncated chunk size prefix")
            csize = int.from_bytes(payload[pos:pos + 2], "big")
            usize = min(MAX_CHUNK_SIZE, target - len(out))
            pos += 2
        if pos + csize > len(payload):
            raise DecodeFailed(f"chunk overrun: need {csize}, have {len(payload) - pos}")
        chunk_bytes = payload[pos:pos + csize]
        pos += csize
        out.extend(lz.decompress_next(chunk_bytes, usize))
    return bytes(out[:target])


@lru_cache(maxsize=6)
def load_rsc7(path):
    """Read an RSC resource and return (decoded [virtual][physical] buffer, vsize).

    All headers are little-endian and the payload is the same LZXD stream; only the
    header in front of it differs between the variants seen in the wild:
      RSC7           : magic, version, flags, pad                   -> payload at 16
      RSC\\x05        : magic, version, flags, 0x0FF512F1, size(BE)  -> payload at 20
      RSC7 re-wrapped: RSC7 header followed by the leftover "12 F1 <size>"
                       of an RSC5 header (seen in some re-saved files) -> payload at 22
    The payload offset is detected from the header and, if decoding fails, the other
    offsets are tried."""
    with open(path, "rb") as f:
        data = f.read()
    magic = data[0:4]
    if magic not in (b"RSC7", b"RSC\x05"):
        raise ValueError(f"{path}: not an RSC7/RSC5 file (magic={magic})")
    flags = struct.unpack("<I", data[8:12])[0]
    vsize, psize = rsc7_sizes(flags)
    target = vsize + psize

    if data[12:16] == b"\x0f\xf5\x12\xf1":
        primary = 20
    elif data[16:18] == b"\x12\xf1":
        primary = 22
    else:
        primary = 16
    offsets = [primary] + [o for o in (16, 20, 22) if o != primary]

    last_err = None
    for off in offsets:
        payload = data[off:]
        for win in ("KB128", "KB64", "KB32", "KB256", "KB512", "MB1"):
            try:
                decoded = decode_xmem_lzxd(payload, target, win)
                return decoded, vsize
            except DecodeFailed as e:
                last_err = e
    raise RuntimeError(f"{path}: failed to decompress ({last_err})")


# ============================================================================
# RSC5 geometry (drawable -> lod -> model -> geometry)
# ============================================================================

class R:
    """Big-endian RSC5 buffer reader with virtual/physical pointer resolution."""

    def __init__(self, data: bytes, vsize: int):
        self.data = data
        self.vsize = vsize

    def addr_to_offset(self, addr: int):
        if addr == 0:
            return None
        top = (addr >> 28) & 0xF
        low = addr & 0x0FFFFFFF
        if top == 5:
            return low
        elif top == 6:
            return low + self.vsize
        else:
            raise ValueError(f"bad pointer 0x{addr:08x}")

    def u8(self, o): return self.data[o]
    def u16(self, o): return struct.unpack_from(">H", self.data, o)[0]
    def u32(self, o): return struct.unpack_from(">I", self.data, o)[0]
    def u64(self, o): return struct.unpack_from(">Q", self.data, o)[0]
    def f32(self, o): return struct.unpack_from(">f", self.data, o)[0]
    def half(self, o): return struct.unpack_from(">e", self.data, o)[0]

    def cstr(self, o):
        end = self.data.index(b"\x00", o)
        return self.data[o:end].decode("utf-8", errors="replace")

    def str_ptr(self, off):
        addr = self.u32(off)
        o = self.addr_to_offset(addr)
        return self.cstr(o) if o is not None else None


# ---------------------------------------------------------------------------
# vertex component decode
# ---------------------------------------------------------------------------

COMP_SIZE = {0: 0, 1: 4, 2: 4, 3: 8, 4: 0, 5: 8, 6: 12, 7: 16, 8: 4, 9: 4, 10: 4}
COMP_COUNT = {0: 0, 1: 2, 2: 1, 3: 4, 4: 0, 5: 2, 6: 3, 7: 4, 8: 4, 9: 4, 10: 3}
SEMANTIC_FOR_INDEX = {0: "POSITION", 1: "BLENDWEIGHTS", 2: "BLENDINDICES", 3: "NORMAL",
                      4: "COLOR", 5: "COLOR", 14: "TANGENT", 15: "BINORMAL"}
for _i in range(6, 14):
    SEMANTIC_FOR_INDEX[_i] = "TEXCOORD"


def decode_dec3n(u32val):
    ux = u32val & 0x3FF
    uy = (u32val >> 10) & 0x3FF
    uz = (u32val >> 20) & 0x3FF

    def dec(v):
        sign = (v >> 9) & 1
        rem = v & 0x1FF
        if sign == 0:
            return rem / 511.0
        else:
            return -((511 - rem) / 511.0)
    return dec(ux), dec(uy), dec(uz)


class VertexDecl:
    def __init__(self, r: R, off):
        self.flags = r.u32(off)
        self.stride = r.u16(off + 4)
        self.count = r.u8(off + 6)
        # u8 at off+7 unused/padding here (matches C# layout: Flags4,Stride2,Unk1,Count1,Types8)
        self.types = r.u64(off + 8)
        self.components = []  # list of (bit_index, comp_type, byte_offset, semantic)
        byte_off = 0
        for i in range(16):
            if (self.flags >> i) & 1:
                ctype = (self.types >> (i * 4)) & 0xF
                sem = SEMANTIC_FOR_INDEX.get(i, f"UNK{i}")
                self.components.append((i, ctype, byte_off, sem))
                byte_off += COMP_SIZE.get(ctype, 0)

    def decode_vertex(self, r: R, vbase):
        out = {}
        for bit_index, ctype, coff, sem in self.components:
            o = vbase + coff
            if ctype == 2:  # Float
                val = r.f32(o)
            elif ctype == 5:  # Float2
                val = (r.f32(o), r.f32(o + 4))
            elif ctype == 6:  # Float3
                val = (r.f32(o), r.f32(o + 4), r.f32(o + 8))
            elif ctype == 7:  # Float4
                val = (r.f32(o), r.f32(o + 4), r.f32(o + 8), r.f32(o + 12))
            elif ctype == 1:  # Half2
                val = (r.half(o), r.half(o + 2))
            elif ctype == 3:  # Half4
                val = (r.half(o), r.half(o + 2), r.half(o + 4), r.half(o + 6))
            elif ctype == 10:  # Dec3N
                val = decode_dec3n(r.u32(o))
            elif ctype in (8, 9):  # UByte4 / Colour
                val = (r.u8(o), r.u8(o + 1), r.u8(o + 2), r.u8(o + 3))
            else:
                val = None
            key = (sem, bit_index)
            out[key] = val
        return out


class Geometry:
    def __init__(self):
        self.positions = []
        self.normals = []
        self.uvs = []
        self.indices = []
        self.shader_id = 0
        self.name = None


def parse_index_buffer(r: R, off):
    indices_count = r.u32(off + 4)
    indices_ptr = r.u32(off + 8)
    ioff = r.addr_to_offset(indices_ptr)
    indices = []
    if ioff is not None:
        for i in range(indices_count):
            indices.append(r.u16(ioff + i * 2))
    return indices


def parse_vertex_buffer(r: R, off):
    vertex_stride = r.u16(off + 4)
    data1_ptr = r.u32(off + 8)
    vertex_count = r.u32(off + 12)
    data2_ptr = r.u32(off + 16)
    # Unknown_28h(4) then Layout ptr at off+24
    layout_ptr = r.u32(off + 24)

    data_ptr = data1_ptr or data2_ptr
    doff = r.addr_to_offset(data_ptr)

    layout = None
    loff = r.addr_to_offset(layout_ptr)
    if loff is not None:
        layout = VertexDecl(r, loff)

    return vertex_count, vertex_stride, doff, layout


def parse_geometry(r: R, off):
    vb_ptr = r.u32(off + 12)          # VertexBuffer ptr
    ib_ptr = r.u32(off + 28)          # IndexBuffer ptr
    indices_count = r.u32(off + 44)
    vertex_count = r.u16(off + 52)

    geo = Geometry()

    vboff = r.addr_to_offset(vb_ptr)
    if vboff is not None:
        vcount, stride, doff, layout = parse_vertex_buffer(r, vboff)
        if doff is not None and layout is not None:
            for vi in range(vcount):
                vbase = doff + vi * stride
                comps = layout.decode_vertex(r, vbase)
                pos = comps.get(("POSITION", 0))
                nrm = None
                uv = None
                for (sem, bidx), val in comps.items():
                    if sem == "NORMAL" and nrm is None:
                        nrm = val
                    if sem == "TEXCOORD" and uv is None:
                        uv = val
                geo.positions.append(pos if pos else (0.0, 0.0, 0.0))
                geo.normals.append(nrm if nrm else (0.0, 0.0, 1.0))
                geo.uvs.append((uv[0], uv[1]) if uv else (0.0, 0.0))

    iboff = r.addr_to_offset(ib_ptr)
    if iboff is not None:
        geo.indices = parse_index_buffer(r, iboff)

    return geo


def parse_model(r: R, off):
    """Returns list of Geometry for this Rsc5DrawableModel."""
    geoms_pos = r.u32(off + 4)
    geoms_cnt = r.u16(off + 8)
    goff = r.addr_to_offset(geoms_pos)
    geoms = []
    if goff is not None:
        ptrs = [r.u32(goff + i * 4) for i in range(geoms_cnt)]
        for p in ptrs:
            po = r.addr_to_offset(p)
            if po is not None:
                geoms.append(parse_geometry(r, po))
    return geoms


def parse_lod(r: R, off):
    """Rsc5DrawableLod: just a Rsc5PtrArr<Rsc5DrawableModel> (Position/Count/Cap at start)."""
    pos = r.u32(off)
    cnt = r.u16(off + 4)
    moff = r.addr_to_offset(pos)
    models = []
    if moff is not None:
        ptrs = [r.u32(moff + i * 4) for i in range(cnt)]
        for p in ptrs:
            po = r.addr_to_offset(p)
            if po is not None:
                models.append(po)
    return models


def parse_drawable(r: R, off, verbose=False):
    """Rsc5DrawableBase layout (offsets relative to `off`):
      0x00 VFT
      0x04 BlockMap ptr
      0x08 ShaderGroup ptr
      0x0C SkeletonPtr
      0x10 BoundingCenter (3xf32)
      0x1C Unknown
      0x20 BoundingBoxMin (3xf32)
      0x2C Unknown
      0x30 BoundingBoxMax (3xf32)
      0x3C Unknown
      0x40 LodHigh ptr
      0x44 LodMed ptr
      0x48 LodLow ptr
      0x4C LodVlow ptr
      ...
    """
    lod_high_ptr = r.u32(off + 0x40)
    all_geoms = []
    lodoff = r.addr_to_offset(lod_high_ptr)
    if lodoff is not None:
        model_offs = parse_lod(r, lodoff)
        if verbose:
            print(f"  LodHigh: {len(model_offs)} models")
        for moff in model_offs:
            geoms = parse_model(r, moff)
            all_geoms.extend(geoms)
    return all_geoms


def parse_drawable_dictionary(r: R, verbose=False):
    """Root object at offset 0: Rsc5DrawableDictionary<Rsc5Drawable>."""
    off = 0
    off += 4  # VFT
    off += 4  # BlockMap ptr
    off += 4  # Unknown_10h
    off += 4  # Unknown_18h
    hashes_pos = r.u32(off); off += 8  # Rsc5Arr<JenkHash> header
    draw_pos = r.u32(off); draw_cnt = r.u16(off + 4); off += 8

    doff = r.addr_to_offset(draw_pos)
    results = []
    if doff is not None:
        ptrs = [r.u32(doff + i * 4) for i in range(draw_cnt)]
        for p in ptrs:
            po = r.addr_to_offset(p)
            if po is not None:
                geoms = parse_drawable(r, po, verbose=verbose)
                results.append(geoms)
    return results


# ============================================================================
# Textures (.xtd): Xbox 360 untiling + BC1/BC2/BC3 decode
# ============================================================================

FMT_DXT1 = 0x12
FMT_DXT3 = 0x13
FMT_DXT5 = 0x14
FMT_DXT5A = 0x3B   # single channel, BC4-like (specular / env maps)
SUPPORTED_FORMATS = (FMT_DXT1, FMT_DXT3, FMT_DXT5, FMT_DXT5A)


# ---------------------------------------------------------------- untiling
def _tiled_x(offset, width, pitch):
    aw = (width + 31) & ~31
    lb = (pitch >> 2) + ((pitch >> 1) >> (pitch >> 2))
    ob = offset << lb
    ot = ((ob & ~4095) >> 3) + ((ob & 1792) >> 2) + (ob & 63)
    om = ot >> (7 + lb)
    mx = (om % (aw >> 5)) << 2
    tile = (((ot >> (5 + lb)) & 2) + (ob >> 6)) & 3
    macro = (mx + tile) << 3
    micro = ((((ot >> 1) & ~15) + (ot & 15)) & ((pitch << 3) - 1)) >> lb
    return macro + micro


def _tiled_y(offset, width, pitch):
    aw = (width + 31) & ~31
    lb = (pitch >> 2) + ((pitch >> 1) >> (pitch >> 2))
    ob = offset << lb
    ot = ((ob & ~4095) >> 3) + ((ob & 1792) >> 2) + (ob & 63)
    om = ot >> (7 + lb)
    my = (om // (aw >> 5)) << 2
    tile = ((ot >> (6 + lb)) & 1) + ((ob & 2048) >> 10)
    macro = (my + tile) << 3
    micro = ((((ot & (((pitch << 6) - 1) & ~31)) + ((ot & 15) << 1)) >> (3 + lb)) & ~1)
    return macro + micro + ((ot & 16) >> 4)


_map_cache = {}


def _tile_map(aw, ah, pitch):
    key = (aw, ah, pitch)
    if key not in _map_cache:
        n = aw * ah
        xs = np.empty(n, dtype=np.int32)
        ys = np.empty(n, dtype=np.int32)
        for k in range(n):
            xs[k] = _tiled_x(k, aw, pitch)
            ys[k] = _tiled_y(k, aw, pitch)
        _map_cache[key] = (xs, ys)
    return _map_cache[key]


def swap16(b: bytes) -> bytes:
    a = np.frombuffer(b, dtype=np.uint8).copy()
    a[0::2], a[1::2] = a[1::2].copy(), a[0::2].copy()
    return a.tobytes()


def untile_blocks(src: bytes, w_px, h_px, pitch):
    """Return linear block bytes (bw*bh*pitch)."""
    bw = max(w_px // 4, 1)
    bh = max(h_px // 4, 1)
    aw = (bw + 31) & ~31
    ah = (bh + 31) & ~31
    xs, ys = _tile_map(aw, ah, pitch)
    n = aw * ah
    blocks = np.frombuffer(src[:n * pitch].ljust(n * pitch, b"\0"),
                           dtype=np.uint8).reshape(n, pitch)
    sel = (xs < bw) & (ys < bh)
    out = np.zeros((bh, bw, pitch), dtype=np.uint8)
    out[ys[sel], xs[sel]] = blocks[sel]
    return out.reshape(-1).tobytes()


# ------------------------------------------------------------ DXT decoding
def _rgb565(c):
    r = ((c >> 11) & 31).astype(np.int32)
    g = ((c >> 5) & 63).astype(np.int32)
    b = (c & 31).astype(np.int32)
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1)


def _decode_color(cb, w, h, allow_alpha):
    """cb: (N,8) uint8 -> (N,16,4) rgba"""
    c0 = cb[:, 0].astype(np.uint16) | (cb[:, 1].astype(np.uint16) << 8)
    c1 = cb[:, 2].astype(np.uint16) | (cb[:, 3].astype(np.uint16) << 8)
    idx = (cb[:, 4].astype(np.uint32) | (cb[:, 5].astype(np.uint32) << 8) |
           (cb[:, 6].astype(np.uint32) << 16) | (cb[:, 7].astype(np.uint32) << 24))
    p0, p1 = _rgb565(c0), _rgb565(c1)
    four = (c0 > c1) | (not allow_alpha)
    p2 = np.where(four[:, None], (2 * p0 + p1) // 3, (p0 + p1) // 2)
    p3 = np.where(four[:, None], (p0 + 2 * p1) // 3, np.zeros_like(p0))
    pal = np.stack([p0, p1, p2, p3], 1)  # N,4,3
    sh = np.arange(16, dtype=np.uint32) * 2
    ci = ((idx[:, None] >> sh[None, :]) & 3).astype(np.int32)  # N,16
    rgb = np.take_along_axis(pal, ci[:, :, None].repeat(3, 2), 1)
    a = np.full((len(cb), 16, 1), 255, dtype=np.int32)
    if allow_alpha:
        a = np.where(((c0 <= c1)[:, None] & (ci == 3))[:, :, None], 0, a)
    return np.concatenate([rgb, a], -1)


def _decode_alpha_blocks(ab):
    """ab: (n, 8) uint8 DXT5-style alpha block -> (n, 16) int32 values."""
    n = len(ab)
    a0 = ab[:, 0].astype(np.int32)
    a1 = ab[:, 1].astype(np.int32)
    bits = np.zeros(n, dtype=np.uint64)
    for i in range(6):
        bits |= ab[:, 2 + i].astype(np.uint64) << np.uint64(8 * i)
    sh = (np.arange(16, dtype=np.uint64) * np.uint64(3))
    ai = ((bits[:, None] >> sh[None, :]) & np.uint64(7)).astype(np.int32)
    pal = np.zeros((n, 8), dtype=np.int32)
    pal[:, 0], pal[:, 1] = a0, a1
    gt = a0 > a1
    for k in range(1, 7):
        pal[:, k + 1] = np.where(gt, ((7 - k) * a0 + k * a1) // 7,
                                 np.where(k <= 4, ((5 - k) * a0 + k * a1) // 5, 0))
    pal[:, 7] = np.where(gt, pal[:, 7], 255)
    return np.take_along_axis(pal, ai, 1)


def decode_bc(data: bytes, w, h, fmt):
    bw, bh = max(w // 4, 1), max(h // 4, 1)
    n = bw * bh
    if fmt == FMT_DXT1:
        blocks = np.frombuffer(data[:n * 8], np.uint8).reshape(n, 8)
        px = _decode_color(blocks, w, h, True)
    elif fmt == FMT_DXT5A:
        blocks = np.frombuffer(data[:n * 8], np.uint8).reshape(n, 8)
        v = _decode_alpha_blocks(blocks)
        px = np.empty((n, 16, 4), dtype=np.int32)
        px[:, :, 0] = px[:, :, 1] = px[:, :, 2] = v
        px[:, :, 3] = 255
    else:
        blocks = np.frombuffer(data[:n * 16], np.uint8).reshape(n, 16)
        px = _decode_color(blocks[:, 8:], w, h, False)
        if fmt == FMT_DXT5:
            alpha = _decode_alpha_blocks(blocks[:, :8])
        else:  # DXT3
            ab = blocks[:, :8]
            v = np.zeros((n, 16), dtype=np.int32)
            for i in range(8):
                v[:, 2 * i] = (ab[:, i] & 15) * 17
                v[:, 2 * i + 1] = (ab[:, i] >> 4) * 17
            alpha = v
        px[:, :, 3] = alpha
    img = px.reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return Image.fromarray(img.astype(np.uint8), "RGBA")


# ------------------------------------------------------------- xtd parsing
class Tex:
    pass


def _clean_tex_name(raw):
    return raw.replace("pack:/", "").replace(".dds", "").lower()


_NORMAL_NAME = re.compile(r"(_nm|_n|_normal)(_\d+)?$")


def fix_two_channel_normal(name, img):
    """Some normal maps store only X and Y (blue is 0 everywhere) and the game rebuilds
    Z in the shader. Viewers expect a full RGB normal map, so rebuild Z = sqrt(1-x^2-y^2)."""
    if not _NORMAL_NAME.search(name):
        return img
    a = np.asarray(img).copy()
    if a[..., 2].max() > 8:
        return img
    x = a[..., 0].astype(np.float32) / 127.5 - 1.0
    y = a[..., 1].astype(np.float32) / 127.5 - 1.0
    z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
    a[..., 2] = np.clip((z * 0.5 + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(a, "RGBA")


def parse_texture_dict(d, v, root=0, only=None, names_only=False):
    """Parse a Rsc5TextureDictionary that starts at buffer offset `root` of the decoded
    buffer `d` (virtual size `v`). `only` (set of names) limits which textures are
    decoded; `names_only` skips decoding entirely."""
    u32 = lambda o: struct.unpack_from(">I", d, o)[0]
    u16 = lambda o: struct.unpack_from(">H", d, o)[0]

    def a2o(a):
        return (a & 0x0FFFFFFF) if (a >> 28) == 5 else (a & 0x0FFFFFFF) + v

    arr = a2o(u32(root + 24))
    cnt = u16(root + 28)
    texs = []
    for i in range(cnt):
        h = a2o(u32(arr + i * 4))
        t = Tex()
        nptr = u32(h + 0x18)
        raw = d[a2o(nptr):].split(b"\0")[0].decode(errors="replace") if nptr else ""
        t.name = _clean_tex_name(raw)
        t.image = None
        t.error = None
        if names_only or (only is not None and t.name not in only):
            t.error = "not requested"
            texs.append(t)
            continue
        try:
            desc = a2o(u32(h + 0x1C)) + 0x20
            p1, a = u32(desc), u32(desc + 4)
            if (p1 >> 28) != 6:
                raise ValueError("bad descriptor")
            t.w = (a & 0x1FFF) + 1
            t.h = ((a >> 13) & 0x1FFF) + 1
            t.fmt = p1 & 0x3F
            if t.fmt not in SUPPORTED_FORMATS:
                raise ValueError(f"unsupported format 0x{t.fmt:x}")
            pitch = 16 if t.fmt in (FMT_DXT3, FMT_DXT5) else 8
            bw, bh = max(t.w // 4, 1), max(t.h // 4, 1)
            aw, ah = (bw + 31) & ~31, (bh + 31) & ~31
            base = (p1 & 0x0FFFF000) + v
            if base + aw * ah * pitch > len(d):
                raise ValueError("texture data outside the resource")
            src = swap16(d[base:base + aw * ah * pitch])
            lin = untile_blocks(src, t.w, t.h, pitch)
            t.image = fix_two_channel_normal(t.name, decode_bc(lin, t.w, t.h, t.fmt))
        except Exception as e:  # noqa
            t.error = str(e)
        texs.append(t)
    return texs


def parse_xtd(path, only=None):
    """Texture dictionary file (.xtd): the dictionary is the root object."""
    d, v = load_rsc7(path)
    return parse_texture_dict(d, v, 0, only)


_NAMES_CACHE = {}


def xtd_names(path):
    """Set of texture names in an .xtd (no pixel decoding; cached)."""
    key = os.path.abspath(path)
    if key not in _NAMES_CACHE:
        try:
            d, v = load_rsc7(path)
            _NAMES_CACHE[key] = {t.name for t in parse_texture_dict(d, v, 0, names_only=True)}
        except Exception:
            _NAMES_CACHE[key] = set()
    return _NAMES_CACHE[key]


def embedded_dict_offset(r, drawable_off):
    """Offset of the texture dictionary embedded in a drawable's shader group
    (many props carry their own textures), or None."""
    sg = r.addr_to_offset(r.u32(drawable_off + 8))
    if sg is None:
        return None
    return r.addr_to_offset(r.u32(sg + 4))


def parse_embedded_textures(r, drawable_off, only=None):
    root = embedded_dict_offset(r, drawable_off)
    if root is None:
        return []
    try:
        return parse_texture_dict(r.data, r.vsize, root, only)
    except Exception:
        return []


# ============================================================================
# Shared texture dictionaries
# ============================================================================
def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def load_textures(own_xtd, shared, wanted, embedded=(), model_stem=""):
    """Return ({name: Tex}, borrowed) for the textures a model references.

    A model may combine several sources; each one only supplies names the previous
    ones lacked:
      1. the model's own .xtd (same file name) - everything in it is loaded;
      2. the texture dictionaries embedded in the model itself;
      3. --tex-dir dictionaries;
      4. shared texture archives: .xtd files that belong to no model (`shared[0]`).
         The archive covering the most missing names wins, ties go to the closest file
         name, then to the model's own folder.
    .xtd files that belong to another model (e.g. another ped's dictionary) are never
    candidates for step 4: peds reuse texture names with different content.
    Each Tex gets .origin = 'own' | 'embedded' | <path of the .xtd it came from>.
    `borrowed` lists (name, source) for every texture that did not come from own."""
    siblings, explicit = shared
    textures = {}
    if own_xtd and os.path.exists(own_xtd):
        for t in parse_xtd(own_xtd):
            t.origin = "own"
            textures[t.name] = t

    def ok(n):
        return n in textures and textures[n].image is not None

    wanted = {n for n in wanted if n}
    missing = {n for n in wanted if not ok(n)}
    borrowed = []

    for r, d_off in embedded:
        if not missing:
            break
        for t in parse_embedded_textures(r, d_off, only=missing):
            if t.name in missing and t.image is not None:
                t.origin = "embedded"
                textures[t.name] = t
                missing.discard(t.name)
                borrowed.append((t.name, "embedded"))

    def take(path):
        try:
            found = parse_xtd(path, only=missing)
        except Exception:
            return 0
        got = 0
        for t in found:
            if t.name in missing and t.image is not None:
                t.origin = path
                textures[t.name] = t
                missing.discard(t.name)
                borrowed.append((t.name, os.path.basename(path)))
                got += 1
        return got

    own_abs = os.path.abspath(own_xtd) if own_xtd else None
    for path in explicit:
        if not missing:
            break
        if os.path.abspath(path) != own_abs and xtd_names(path) & missing:
            take(path)

    cands = [p for p in siblings if os.path.abspath(p) != own_abs and p not in explicit]
    order = {p: i for i, p in enumerate(cands)}
    while missing and cands:
        def key(p):
            stem = os.path.splitext(os.path.basename(p))[0].lower()
            return (-len(xtd_names(p) & missing), -_common_prefix(stem, model_stem.lower()),
                    order[p])
        best = min(cands, key=key)
        if not (xtd_names(best) & missing):
            break
        take(best)
        cands.remove(best)
    return textures, borrowed


def report_borrowed(borrowed):
    if not borrowed:
        return
    counts = {}
    for _, src in borrowed:
        counts[src] = counts.get(src, 0) + 1
    print("    textures from: " + ", ".join(f"{k} ({v})" for k, v in sorted(counts.items())))


def safe_name(name):
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)


def make_tex_saver(out_dir, base, textures, skipped=None):
    """Returns (save_tex(name) -> relative png path or None, saved dict).

    Textures of the model's own .xtd / embedded dictionary go to <model>_textures/;
    textures taken from a shared .xtd go to <thatxtd>_textures/, written once and
    reused by every model that needs them."""
    saved = {}

    def save_tex(name):
        if not name:
            return None
        if name in saved:
            return saved[name]
        t = textures.get(name)
        if t is None or t.image is None:
            if skipped is not None:
                skipped.append((name, "missing" if t is None else t.error))
            saved[name] = None
            return None
        origin = getattr(t, "origin", "own")
        if origin in ("own", "embedded"):
            sub, overwrite = base + "_textures", True
        else:
            sub, overwrite = os.path.splitext(os.path.basename(origin))[0] + "_textures", False
        tdir = os.path.join(out_dir, sub)
        os.makedirs(tdir, exist_ok=True)
        fn = safe_name(name) + ".png"
        path = os.path.join(tdir, fn)
        if overwrite or not os.path.exists(path):
            t.image.save(path)
        saved[name] = f"{sub}/{fn}"
        return saved[name]

    return save_tex, saved


def has_real_alpha(textures, name):
    """True when a texture's alpha channel really varies (glass, decals, cut-outs).
    In these shaders diffuse alpha is the opacity ('dm'+'tm' packed together)."""
    t = textures.get(name) if name else None
    if t is None or t.image is None:
        return False
    return int(np.asarray(t.image)[..., 3].min()) < 250


def mtl_block(mname, textures, save_tex, diff, nrm, spc):
    """Text of one .mtl material entry."""
    lines = [f"newmtl {mname}", "Ka 1 1 1", "Kd 1 1 1", "Ks 0.2 0.2 0.2", "Ns 20"]
    p = save_tex(diff)
    if p:
        lines.append(f"map_Kd {p}")
        if has_real_alpha(textures, diff):
            lines += ["d 1.0", f"map_d {p}"]
    for key, tn in (("map_Bump", nrm), ("map_Ks", spc)):
        p = save_tex(tn)
        if p:
            lines.append(f"{key} {p}")
    return "\n".join(lines) + "\n\n"


def missing_note(skipped):
    names = sorted({n for n, _ in skipped})
    if not names:
        return ""
    shown = ", ".join(names[:4]) + (", ..." if len(names) > 4 else "")
    return f", {len(names)} texture(s) not found ({shown})"


# ============================================================================
# Shaders / materials + ped export
# ============================================================================

H_DIFFUSE = 0xF1FE2B71
H_NORMAL = 0x46B7C64F
H_SPEC = 0x608799C6


def parse_shader_group(r: R, drawable_off):
    """Returns list of shaders; each shader = {param_hash: texture_name}."""
    sg_ptr = r.u32(drawable_off + 8)
    sg = r.addr_to_offset(sg_ptr)
    if sg is None:
        return []
    shaders_pos = r.u32(sg + 8)
    count = r.u16(sg + 12)
    so = r.addr_to_offset(shaders_pos)
    out = []
    for j in range(count):
        fx = r.addr_to_offset(r.u32(so + 4 * j))
        pcount = r.u8(fx + 8)
        base = r.addr_to_offset(r.u32(fx))
        texmap = {}
        for k in range(pcount):
            h = r.u32(base + 12 * k)
            dtype = r.u8(base + 12 * k + 4)
            dptr = r.u32(base + 12 * k + 8)
            if dtype == 0 and dptr:
                t = r.addr_to_offset(dptr)
                nptr = r.u32(t + 0x18)
                if nptr:
                    nm = r.cstr(r.addr_to_offset(nptr))
                    texmap[h] = nm.replace("pack:/", "").replace(".dds", "").lower()
        out.append(texmap)
    return out


def drawable_geoms(r, d_off):
    """Geometries (LOD 0) of one drawable at d_off, with shader texture names."""
    shaders = parse_shader_group(r, d_off)
    lod = r.addr_to_offset(r.u32(d_off + 0x40))
    geoms = []
    if lod is not None:
        for moff in parse_lod(r, lod):
            cnt = r.u16(moff + 8)
            smap_ptr = r.addr_to_offset(r.u32(moff + 16))
            gpos = r.addr_to_offset(r.u32(moff + 4))
            for gi in range(cnt):
                g = parse_geometry(r, r.addr_to_offset(r.u32(gpos + 4 * gi)))
                g.shader_id = r.u16(smap_ptr + 2 * gi) if smap_ptr is not None else 0
                g.textures = shaders[g.shader_id] if g.shader_id < len(shaders) else {}
                geoms.append(g)
    return geoms


def load_drawables(path):
    """(reader, [drawable offsets]) for .xdd (dictionary) or .xdr (single drawable)."""
    data, vsize = load_rsc7(path)
    r = R(data, vsize)
    if os.path.splitext(path)[1].lower() == ".xdr":
        return r, [0]
    dp = r.u32(24)
    dc = r.u16(28)
    doff = r.addr_to_offset(dp)
    return r, [r.addr_to_offset(r.u32(doff + 4 * i)) for i in range(dc)]


def parse_drawables(path):
    """.xdd = drawable dictionary (root at 0 holds N drawables),
    .xdr = a single drawable (root at 0 is the drawable itself)."""
    r, offs = load_drawables(path)
    return [drawable_geoms(r, o) for o in offs]


def export_ped(xdd_path, xtd_path, out_dir, shared=((), ())):
    base = os.path.splitext(os.path.basename(xdd_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    r, offs = load_drawables(xdd_path)
    drawables = [drawable_geoms(r, o) for o in offs]
    wanted = {n for geoms in drawables for g in geoms for n in
              (g.textures.get(H_DIFFUSE), g.textures.get(H_NORMAL), g.textures.get(H_SPEC))}
    textures, borrowed = load_textures(xtd_path, shared, wanted,
                                       [(r, o) for o in offs], base)
    report_borrowed(borrowed)
    skipped = []
    save_tex, saved = make_tex_saver(out_dir, base, textures, skipped)

    mats = {}
    with open(os.path.join(out_dir, base + ".obj"), "w") as f:
        f.write(f"# Max Payne 3 (X360) ped: {base}\nmtllib {base}.mtl\n")
        vbase = 1
        n = 0
        for di, geoms in enumerate(drawables):
            for gi, g in enumerate(geoms):
                diff = g.textures.get(H_DIFFUSE)
                nrm = g.textures.get(H_NORMAL)
                spc = g.textures.get(H_SPEC)
                mname = diff or f"mat_{di}_{gi}"
                mats[mname] = (diff, nrm, spc)
                f.write(f"o part_{di}_{gi}\nusemtl {mname}\n")
                for p in g.positions:
                    f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                for nn in g.normals:
                    f.write(f"vn {nn[0]:.6f} {nn[1]:.6f} {nn[2]:.6f}\n")
                for uv in g.uvs:
                    f.write(f"vt {uv[0]:.6f} {1.0 - uv[1]:.6f}\n")
                idx = g.indices
                for t in range(0, len(idx) - 2, 3):
                    a, b, c = idx[t] + vbase, idx[t + 1] + vbase, idx[t + 2] + vbase
                    if a == b or b == c or a == c:
                        continue
                    f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
                vbase += len(g.positions)
                n += 1

    with open(os.path.join(out_dir, base + ".mtl"), "w") as f:
        for mname, (diff, nrm, spc) in mats.items():
            f.write(mtl_block(mname, textures, save_tex, diff, nrm, spc))

    # also dump every texture in the dictionary, even unreferenced ones
    for name in textures:
        save_tex(name)

    ok = sum(1 for v in saved.values() if v)
    print(f"{base}: {n} meshes, {len(mats)} materials, {ok} textures written"
          + missing_note(skipped))


# ============================================================================
# Vehicle export (.xft fragment)
# ============================================================================

WHEEL_BONES = (33, 34, 35, 36)


def read_geoms(r, drawable_off, lod_index=0):
    lod_ptr = r.u32(drawable_off + 0x40 + 4 * lod_index)
    if not lod_ptr:
        return []
    out = []
    for moff in parse_lod(r, r.addr_to_offset(lod_ptr)):
        cnt = r.u16(moff + 8)
        gpos = r.addr_to_offset(r.u32(moff + 4))
        sm = r.addr_to_offset(r.u32(moff + 16))
        for gi in range(cnt):
            g = parse_geometry(r, r.addr_to_offset(r.u32(gpos + 4 * gi)))
            g.shader_id = r.u16(sm + 2 * gi) if sm is not None else 0
            out.append(g)
    return out


def bone_positions(r, drawable_off):
    """World translation of every bone from the skeleton default transforms."""
    import struct
    import numpy as np
    sk = r.addr_to_offset(r.u32(drawable_off + 0xC))
    n = r.u16(sk + 0x14)
    par = r.addr_to_offset(r.u32(sk + 4))
    dt = r.addr_to_offset(r.u32(sk + 0x10))
    parents = [struct.unpack_from(">i", r.data, par + 4 * i)[0] for i in range(n)]
    parents[0] = -1
    loc = [np.array([[r.f32(dt + 64 * i + 16 * k + 4 * j) for j in range(4)]
                     for k in range(4)]) for i in range(n)]
    memo = {}

    def world(i):
        if i not in memo:
            m = loc[i].copy()
            m[:, 3] = [0, 0, 0, 1]
            memo[i] = m if parents[i] < 0 else m @ world(parents[i])
        return memo[i]

    return [tuple(float(x) for x in world(i)[3, :3]) for i in range(n)]


def parse_fragment(xft_path):
    data, vsize = load_rsc7(xft_path)
    r = R(data, vsize)
    name = r.cstr(r.addr_to_offset(r.u32(0xB0))).replace("pack:/", "")
    body = r.addr_to_offset(r.u32(0xB4))
    # fragment root centre-of-gravity offset: child attachment matrices are
    # relative to it, the body mesh is not
    cg = (r.f32(0x20), r.f32(0x24), r.f32(0x28))
    shaders = parse_shader_group(r, body)
    body_geoms = read_geoms(r, body)
    bones = bone_positions(r, body)

    children = []
    cptr = r.addr_to_offset(r.u32(0xD4))
    i = 0
    while True:
        p = r.u32(cptr + 4 * i)
        if (p >> 28) != 5:
            break
        c = r.addr_to_offset(p)
        bone = r.u16(c + 0xE)
        trans = bones[bone] if bone < len(bones) else (0.0, 0.0, 0.0)
        dptr = r.u32(c + 0x90)
        geoms = read_geoms(r, r.addr_to_offset(dptr)) if dptr else []
        children.append(dict(bone=bone, t=trans, geoms=geoms))
        i += 1
    return name, shaders, body_geoms, children, (r, body)


def export_vehicle(xft_path, xtd_path, out_dir, shared=((), ())):
    name, shaders, body, children, (fr, body_off) = parse_fragment(xft_path)
    base = os.path.splitext(os.path.basename(xft_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    wanted = {n for tex in shaders for n in
              (tex.get(H_DIFFUSE), tex.get(H_NORMAL), tex.get(H_SPEC))}
    textures, borrowed = load_textures(xtd_path, shared, wanted, [(fr, body_off)], base)
    report_borrowed(borrowed)
    skipped = []
    save_tex, saved = make_tex_saver(out_dir, base, textures, skipped)

    # wheel model (one stored, instanced on the wheel bones)
    wheel_geoms = next((c["geoms"] for c in children if c["geoms"]), [])
    body_min_z = min((p[2] for g in body for p in g.positions), default=0.0)
    radius = max((p[2] for g in wheel_geoms for p in g.positions), default=0.43)
    wheel_pos = [(c["t"][0], c["t"][1], body_min_z + radius)
                 for c in children if c["bone"] in WHEEL_BONES]

    parts = []  # (label, geom, offset, mirror)
    for i, g in enumerate(body):
        parts.append((f"body_{i}", g, (0, 0, 0), False))
    for wi, t in enumerate(wheel_pos):
        for gi, g in enumerate(wheel_geoms):
            parts.append((f"wheel_{wi}_{gi}", g, t, t[0] > 0))

    mats = {}
    with open(os.path.join(out_dir, base + ".obj"), "w") as f:
        f.write(f"# Max Payne 3 (X360) vehicle: {name}\nmtllib {base}.mtl\n")
        vbase = 1
        for label, g, off, mirror in parts:
            tex = shaders[g.shader_id] if g.shader_id < len(shaders) else {}
            diff = tex.get(H_DIFFUSE)
            mname = diff or f"shader_{g.shader_id}"
            mats[mname] = (diff, tex.get(H_NORMAL), tex.get(H_SPEC))
            f.write(f"o {label}\nusemtl {mname}\n")
            for p in g.positions:
                x, y, z = p
                if mirror:
                    x = -x
                f.write(f"v {x + off[0]:.6f} {y + off[1]:.6f} {z + off[2]:.6f}\n")
            for n in g.normals:
                x = -n[0] if mirror else n[0]
                f.write(f"vn {x:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            for uv in g.uvs:
                f.write(f"vt {uv[0]:.6f} {1.0 - uv[1]:.6f}\n")
            idx = g.indices
            for t in range(0, len(idx) - 2, 3):
                a, b, c = idx[t] + vbase, idx[t + 1] + vbase, idx[t + 2] + vbase
                if a == b or b == c or a == c:
                    continue
                if mirror:
                    b, c = c, b
                f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
            vbase += len(g.positions)

    with open(os.path.join(out_dir, base + ".mtl"), "w") as f:
        for m, (diff, nrm, spc) in mats.items():
            f.write(mtl_block(m, textures, save_tex, diff, nrm, spc))
    for n in textures:
        save_tex(n)

    print(f"{base}: body {len(body)} parts, wheels {len(wheel_pos)}x{len(wheel_geoms)}, "
          f"{len(mats)} materials, {sum(1 for v in saved.values() if v)} textures"
          + missing_note(skipped))


# ---------------------------------------------------------------------------
# Standalone texture dictionary (.xtd without a model)
# ---------------------------------------------------------------------------
def export_textures(xtd_path, out_dir):
    base = os.path.splitext(os.path.basename(xtd_path))[0]
    tex_dir = os.path.join(out_dir, base + "_textures")
    ok = 0
    failed = []
    for t in parse_xtd(xtd_path):
        if t.image is None:
            failed.append((t.name, t.error))
            continue
        os.makedirs(tex_dir, exist_ok=True)
        t.image.save(os.path.join(tex_dir, t.name + ".png"))
        ok += 1
    print(f"{base}: {ok} textures written" + (f", failed {failed}" if failed else ""))
    if ok == 0 and failed:
        raise RuntimeError(f"no texture could be decoded ({failed[0][1]})")


# ---------------------------------------------------------------------------
# Folder processing
# ---------------------------------------------------------------------------
MODEL_EXTS = (".xdd", ".xdr", ".xft")            # produce OBJ (+ textures)
TEXTURE_EXTS = (".xtd",)                         # produce PNGs only (when alone)
KNOWN_UNSUPPORTED = {
    ".xbn": "collision bounds",
    ".xbd": "collision bounds dictionary",
    ".xpl": "map placement data",
    ".xvd": "not a mesh format",
    ".ide": "item definitions (text)",
}


def find_texture(dirpath, stem, extra_dirs):
    """Same-named .xtd next to the model, else in any --tex-dir."""
    for d in [dirpath] + list(extra_dirs):
        for name in (stem + ".xtd", stem + ".XTD"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return None


def list_xtds(dirs):
    out = []
    for d in dirs:
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        out += [os.path.join(d, n) for n in names if n.lower().endswith(".xtd")]
    return out


def tree_xtds(root, recursive=True):
    """Every .xtd under `root` (candidates for textures a model lacks)."""
    if not recursive:
        return list_xtds([root])
    out = []
    for dp, dn, fn in os.walk(root):
        dn.sort()
        out += [os.path.join(dp, n) for n in sorted(fn) if n.lower().endswith(".xtd")]
    return out


MODEL_STEMS_EXT = (".xdd", ".xdr", ".xft")


def model_stems(paths):
    """Lower-case stems of all model files - a .xtd with one of these stems belongs
    to that model and is not a shared archive."""
    return {os.path.splitext(os.path.basename(p))[0].lower() for p in paths
            if p.lower().endswith(MODEL_STEMS_EXT)}


def shared_xtds(dirpath, extra_dirs, all_xtds=None, owned=None):
    """(shared texture archives, explicit --tex-dir .xtd files).

    Shared archives are the .xtd files (own folder first, then the rest of the tree)
    that no model claims by name."""
    local = list_xtds([dirpath])
    if owned is None:
        try:
            owned = model_stems(os.listdir(dirpath))
        except OSError:
            owned = set()
    rest = [p for p in (all_xtds or []) if p not in set(local)]
    pool = [p for p in local + rest
            if os.path.splitext(os.path.basename(p))[0].lower() not in owned]
    return pool, list_xtds(extra_dirs)


def collect_jobs(root, recursive, tex_dirs, exclude=None):
    """Walk `root` and decide what to do with every file.
    Returns (jobs, skipped) - jobs: (kind, model_path, xtd_path, rel_dir, shared_xtds)."""
    jobs, skipped = [], []
    all_x = [p for p in tree_xtds(root, recursive)
             if not (exclude and os.path.abspath(p).startswith(exclude + os.sep))]
    owned = set()
    for _dp, _dn, _fn in (os.walk(root) if recursive else [(root, [], os.listdir(root))]):
        if exclude and os.path.abspath(_dp).startswith(exclude):
            continue
        owned |= model_stems(_fn)
    walker = os.walk(root) if recursive else [(root, [], os.listdir(root))]
    for dirpath, dirnames, filenames in walker:
        dirnames.sort()
        if exclude:  # never read our own output back in
            dirnames[:] = [d for d in dirnames
                           if os.path.abspath(os.path.join(dirpath, d)) != exclude]
        files = {f.lower(): f for f in filenames if os.path.isfile(os.path.join(dirpath, f))}
        by_stem = {}
        for low, orig in files.items():
            stem, ext = os.path.splitext(low)
            by_stem.setdefault(stem, {})[ext] = os.path.join(dirpath, orig)
        rel = os.path.relpath(dirpath, root)
        rel = "" if rel == "." else rel

        for stem in sorted(by_stem):
            exts = by_stem[stem]
            model = None
            if ".xdd" in exts:
                model = (".xdd", "drawable dictionary")
            elif ".xdr" in exts:
                model = (".xdr", "drawable")
            elif ".xft" in exts:
                model = (".xft", "fragment")
            xtd = exts.get(".xtd") or find_texture(dirpath, stem, tex_dirs)

            if model:
                jobs.append((model[0], exts[model[0]], xtd, rel,
                             shared_xtds(dirpath, tex_dirs, all_x, owned)))
                # a .xft next to an .xdd/.xdr is that model's own fragment (ped)
                if model[0] != ".xft" and ".xft" in exts:
                    skipped.append((exts[".xft"], "companion fragment of " + stem + model[0]))
            elif ".xtd" in exts:
                jobs.append((".xtd", exts[".xtd"], exts[".xtd"], rel, ((), ())))

            for ext, path in exts.items():
                if ext in KNOWN_UNSUPPORTED:
                    skipped.append((path, "unsupported: " + KNOWN_UNSUPPORTED[ext]))
                elif ext not in MODEL_EXTS + TEXTURE_EXTS:
                    skipped.append((path, "unknown format"))
    return jobs, skipped


def run_job(kind, model, xtd, out_dir, shared=((), ())):
    if kind in (".xdd", ".xdr"):
        export_ped(model, xtd, out_dir, shared)
    elif kind == ".xft":
        export_vehicle(model, xtd, out_dir, shared)
    elif kind == ".xtd":
        export_textures(model, out_dir)


def process_folder(root, out_root, recursive=True, tex_dirs=()):
    jobs, skipped = collect_jobs(root, recursive, tex_dirs, os.path.abspath(out_root))
    if not jobs:
        print(f"nothing to convert in {root}")
    done, failed = 0, []
    for i, (kind, model, xtd, rel, shared) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {os.path.join(rel, os.path.basename(model))}")
        try:
            run_job(kind, model, xtd, os.path.join(out_root, rel), shared)
            done += 1
        except Exception as e:  # keep going, report at the end
            failed.append((model, f"{type(e).__name__}: {e}"))
            print(f"    FAILED: {type(e).__name__}: {e}")
    print("\n" + "=" * 60)
    print(f"converted: {done}   failed: {len(failed)}   skipped: {len(skipped)}")
    for p, why in failed:
        print(f"  FAILED  {p}: {why}")
    groups = {}
    for p, why in skipped:
        groups.setdefault(why if not why.startswith("companion") else "companion fragment of a ped", []).append(p)
    for why, paths in groups.items():
        for p in paths[:5]:
            print(f"  skipped {p}: {why}")
        if len(paths) > 5:
            print(f"  ... and {len(paths) - 5} more skipped: {why}")
    return len(failed) == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Max Payne 3 (Xbox 360) resource converter: "
                    ".xdd/.xdr/.xft (+ .xtd) -> OBJ + MTL + PNG, .xtd -> PNG. "
                    "Pass a file or a whole folder.")
    ap.add_argument("input", nargs="?", help="a file (.xdd .xdr .xft .xtd) or a folder to convert")
    ap.add_argument("xtd", nargs="?",
                    help="(single-file mode) matching .xtd, default: same name next to the model")
    ap.add_argument("-o", "--out", default="out", help="output folder (default: out)")
    ap.add_argument("--no-recursive", action="store_true",
                    help="folder mode: do not descend into sub-folders")
    ap.add_argument("--tex-dir", action="append", default=[], metavar="DIR",
                    help="extra folder with shared .xtd files to search for missing textures "
                         "(can be repeated)")
    ap.add_argument("--dir", help=argparse.SUPPRESS)  # old spelling of the folder argument
    a = ap.parse_args()
    a.input = a.input or a.dir
    if not a.input:
        ap.print_help()
        raise SystemExit(2)

    if os.path.isdir(a.input):
        ok = process_folder(a.input, a.out, not a.no_recursive, a.tex_dir)
        raise SystemExit(0 if ok else 1)

    if not os.path.isfile(a.input):
        raise SystemExit(f"not found: {a.input}")
    ext = os.path.splitext(a.input)[1].lower()
    if ext not in MODEL_EXTS + TEXTURE_EXTS:
        hint = KNOWN_UNSUPPORTED.get(ext, "unknown format")
        raise SystemExit(f"unsupported file: {a.input} ({hint})")
    xtd = a.xtd or find_texture(os.path.dirname(a.input) or ".",
                                os.path.splitext(os.path.basename(a.input))[0], a.tex_dir)
    run_job(ext, a.input, xtd, a.out,
            shared_xtds(os.path.dirname(a.input) or ".", a.tex_dir))


if __name__ == "__main__":
    main()
