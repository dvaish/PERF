from dataclasses import dataclass
from typing import Iterator, Optional, List, Iterable
from enum import IntEnum
from more_itertools import chunked
import logging

# RISC-V Psuedoinstructions: https://github.com/riscv-non-isa/riscv-asm-manual/blob/master/riscv-asm.md#pseudoinstructions
branches = [
    # RV64I branches
    "beq",
    "bge",
    "bgeu",
    "blt",
    "bltu",
    "bne",
    # RV64C branches
    "c.beqz",
    "c.bnez",
    # Psuedo instructions
    "beqz",
    "bnez",
    "blez",
    "bgez",
    "bltz",
    "bgtz",
    "bgt",
    "ble",
    "bgtu",
    "bleu",
]
jumps = ["j", "jal", "jr", "jalr", "ret", "call", "c.j", "c.jal", "c.jr", "c.jalr", "tail"]
syscalls = ["ecall", "ebreak", "mret", "sret", "uret"]
control_insts = set(branches + jumps + syscalls)
no_target_insts = set(syscalls + ["jr", "jalr", "c.jr", "c.jalr", "ret"])

# # RISC-V Base ISA custom opcode space
# custom = {
#     0b00_010_11: "CUSTOM_0",
#     0b01_010_11: "CUSTOM_1",
#     0b10_110_11: "CUSTOM_2",
#     0b11_110_11: "CUSTOM_3",
# }

# # funct7 map from Gemmini.h
# # These funct7 codes tell us which Gemmini instruction is being executed from the RoCC instruction
# rocc = {
#     0:   "k_CONFIG",
#     1:   "k_MVIN2",
#     2:   "k_MVIN",
#     3:   "k_MVOUT",
#     4:   "k_COMPUTE_PRELOADED",
#     5:   "k_COMPUTE_ACCUMULATE",
#     6:   "k_PRELOAD",
#     7:   "k_FLUSH",
#     8:   "k_LOOP_WS",
#     9:   "k_LOOP_WS_CONFIG_BOUNDS",
#     10:  "k_LOOP_WS_CONFIG_ADDRS_AB",
#     11:  "k_LOOP_WS_CONFIG_ADDRS_DC",
#     12:  "k_LOOP_WS_CONFIG_STRIDES_AB",
#     13:  "k_LOOP_WS_CONFIG_STRIDES_DC",
#     14:  "k_MVIN3",
#     126: "k_COUNTER",
#     15:  "k_LOOP_CONV_WS",
#     16:  "k_LOOP_CONV_WS_CONFIG_1",
#     17:  "k_LOOP_CONV_WS_CONFIG_2",
#     18:  "k_LOOP_CONV_WS_CONFIG_3",
#     19:  "k_LOOP_CONV_WS_CONFIG_4",
#     20:  "k_LOOP_CONV_WS_CONFIG_5",
#     21:  "k_LOOP_CONV_WS_CONFIG_6",
#     23:  "k_MVOUT_SPAD"
# }




class Op(IntEnum):
    Store = 0
    Load = 1


@dataclass
class SpikeCommitInfo:
    address: int
    data: int
    op: Op


@dataclass
class SpikeTraceEntry:
    pc: int
    # the raw decoded instruction from spike
    inst: int
    decoded_inst: str
    # the absolute dynamic instruction count. [inst_count] is zero-indexed
    inst_count: int
    # if the spike log was collected with --log-commits and this trace entry is a memory operation,
    #   [commit_info] will contain the memory operation
    commit_info: Optional[SpikeCommitInfo] = None 

    def is_control_inst(self) -> bool:
        return self.decoded_inst in control_insts

# [full_commit_log] = True if spike was ran with '-l --log-commits', False if spike is only run with '-l'
def parse_spike_log(log_lines: Iterator[str], full_commit_log: bool) -> Iterator[SpikeTraceEntry]:
    inst_count = 0
    for line in log_lines:
        # Example of first line (regular commit log)
        # core   0: 0x0000000080001a8e (0x00009522) c.add   a0, s0
        s = line.split()
        if s[2][0] == ">":
            continue  # this is a spike-decoded label, ignore it
        pc = int(s[2][2:], 16)
        inst = int(s[3][1:-1], 16)
        decoded_inst = s[4]

        # # Extract the opcode and funct7 to decode Gemmini RoCC instructions
        # opcode = inst % (2 ** 7)            # inst[6:0]
        # funct7 = (inst >> 25) % (2 ** 7)    # inst[31:25]

        # rs1 = (inst >> 15) % (2 ** 5)       # inst[15:19]
        # rs2 = (inst >> 20) % (2 ** 5)       # inst[20:24]
        # rd = (inst >> 7)  % (2 ** 5)       # inst[7:11]

        # if opcode in custom:        # All Gemmini RoCC instructions use XCUSTOM_ACC=3 mapping to custom3 opcode
        #     assert decoded_inst == "unknown", f"Expected custom instruction to be labeled unknown, is {decoded_inst}"
        #     if funct7 in rocc:      # Map the funct7 to the Gemmini function
        #         decoded_inst = rocc[funct7]

        # Ignore spike trace outside DRAM
        if pc < 0x8000_0000:
            if full_commit_log:
                next(log_lines, None)
            continue
        commit_info: Optional[SpikeCommitInfo] = None
        if full_commit_log:
            # If the current line is a valid instruction, then we can be sure the next line
            # will contain the commit info
            line2 = next(log_lines, None)
            # Examples of line2 (only seen in full commit log)

            # Regular instruction (single writeback)
            # core   0: 3 0x0000000080001310 (0x832a) x6  0x0000000080023000
            # <hartid>: <priv>          <PC> <inst> <rd>  <writeback data>

            # Store instruction
            # core   0: 3 0x0000000080001bf4 (0xe11c) mem 0x0000000080002050 0x0000000080002060
            # <hartid>: <priv>          <PC>   <inst>           <store addr>       <store data>

            # Load instruction
            # core   0: 3 0x0000000080000250 (0x638c) x11 0x0000000080001d68 mem 0x0000000080001d90
            # <hartid>: <priv>          <PC>   <inst> <rd>       <load data>            <load addr>
            assert line2 is not None
            s2 = line2.split()
            s2_len = len(s2)
            if s2_len == 8 and s2[5] == "mem":  # store instruction
                commit_info = SpikeCommitInfo(
                    address=int(s2[6][2:], 16), data=int(s2[7][2:], 16), op=Op.Store
                )
            elif s2_len == 9 and s2[7] == "mem":  # load instruction
                commit_info = SpikeCommitInfo(
                    address=int(s2[8][2:], 16), data=int(s2[6][2:], 16), op=Op.Load
                )
        yield SpikeTraceEntry(pc, inst, decoded_inst, inst_count, commit_info)
        inst_count += 1
