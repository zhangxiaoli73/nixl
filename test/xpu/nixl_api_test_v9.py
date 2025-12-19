#!/usr/bin/env python3
"""
NIXL API Performance Benchmark (Corrected & Improved)

Simulates sender/receiver engines transferring KV cache blocks over NIXL.
This version fixes a potential INVALID_PARAM error and improves process synchronization.

NIXL_TELEMETRY_ENABLE=1 python examples/python/nixl_api_test_v9.py --nixl_backend UCX --device-type cuda  --ucx-transport "tcp,cuda_copy" --num-heads 8 --head-size 128 --num-layers 28 --input-tokens 160 --num-iterations 10
"""

import argparse
import logging
import multiprocessing
import os
import time
import uuid
from typing import List, Iterator

import msgspec
import torch
import zmq
import pandas as pd
import pdb
# ==============================================================================
# Configuration
# ==============================================================================

ZMQ_HOST = "127.0.0.1"
ZMQ_BASE_PORT = 15555
GET_META_MSG = b"get_meta_msg"
SHUTDOWN_MSG = b"shutdown_msg"

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(processName)s][%(asctime)s] %(message)s",
)

# Assuming 'nixl_agent' is the class name provided by the library.
from nixl._api import nixl_agent as NixlAgent
from nixl._api import nixl_agent_config
import nixl._bindings


class NixlAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    """Metadata structure exchanged between sender and receiver."""
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    num_blocks: int
    block_len: int

import copy
import numpy as np
from dataclasses import dataclass
from nixl._bindings import nixlXferTelemetry
@dataclass
class NixlKVConnectorStats:
    """Container for transfer performance metrics"""

    def __init__(self):
        # if not self.data:
        #     # Empty container init, no data is passed in.
        self.reset()

    def reset(self):
        # Must be serializable
        self.data: dict[str, list[float]] = {
            "transfer_duration": [],
            "post_duration": [],
            "bytes_transferred": [],
            "num_descriptors": [],
            "num_failed_transfers": [],
            "num_failed_notifications": [],
        }

    def record_transfer(self, res: nixlXferTelemetry):
        # Keep metrics units consistent with rest of the code: time us->s
        self.data["transfer_duration"].append(res.xferDuration / 1e6)
        self.data["post_duration"].append(res.postDuration / 1e6)
        self.data["bytes_transferred"].append(res.totalBytes)
        self.data["num_descriptors"].append(res.descCount)

    def record_failed_transfer(self):
        """Record a failed NIXL transfer operation."""
        self.data["num_failed_transfers"].append(1.0)

    def record_failed_notification(self):
        """Record a failed NIXL notification (send_notif)."""
        self.data["num_failed_notifications"].append(1.0)

    def clone_and_reset(self) -> "NixlKVConnectorStats":
        old = copy.copy(self)
        self.reset()
        return old

    def is_empty(self) -> bool:
        return self.num_successful_transfers == 0

    # def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
    #     if not other.is_empty():
    #         for k, v in other.data.items():
    #             accumulator = self.data[k]
    #             assert isinstance(accumulator, list)
    #             accumulator.extend(v)
    #     return self

    def reduce(self) -> dict[str, int | float]:
        # Compute compact representative stats suitable for CLI logging
        if self.is_empty():
            return {
                "Num successful transfers": 0,
                "Avg xfer time (ms)": 0,
                "P90 xfer time (ms)": 0,
                "Avg post time (ms)": 0,
                "P90 post time (ms)": 0,
                "Avg MB per transfer": 0,
                "Throughput (MB/s)": 0,
                "Avg number of descriptors": 0,
            }

        xfer_time = np.asarray(self.data["transfer_duration"])
        post_time = np.asarray(self.data["post_duration"])
        # Convert to MB for CLI logging.
        mb = np.asarray(self.data["bytes_transferred"]) / 2**20
        descs = np.asarray(self.data["num_descriptors"], dtype=np.uint32)
        n = len(descs)
        assert n == self.num_successful_transfers

        total_mb = mb.sum()
        avg_mb = total_mb / n

        total_time_seconds = xfer_time.sum()
        throughput_mb_s = total_mb / total_time_seconds

        return {
            "Num successful transfers": n,
            "Avg xfer time (ms)": round(xfer_time.mean() * 1e3, 3),
            "P90 xfer time (ms)": round(np.percentile(xfer_time, 90) * 1e3, 3),
            "Avg post time (ms)": round(post_time.mean() * 1e3, 3),
            "P90 post time (ms)": round(np.percentile(post_time, 90) * 1e3, 3),
            "Avg MB per transfer": round(avg_mb, 3),
            "Throughput (MB/s)": round(throughput_mb_s, 3),
            "Avg number of descriptors": round(descs.mean(), 1),
        }

    @property
    def num_successful_transfers(self) -> int:
        return len(self.data["transfer_duration"])

# ==============================================================================
# Helpers
# ==============================================================================

def get_block_desc_ids(num_total_blocks: int, block_ids: Iterator[int]) -> List[int]:
    """
    Maps logical block IDs to the indices of their transfer descriptors.
    """
    return list(block_ids)


def allocate_kv_cache(args: argparse.Namespace, block_len: int, dtype: torch.dtype, device_id: int) -> torch.Tensor:
    """Allocate a KV cache buffer on the given device."""
    total_bytes = args.num_blocks * block_len
    num_elements = total_bytes // dtype.itemsize
    device = f"{args.device_type}:{device_id}"
    logging.info(
        f"Allocating KV cache: {total_bytes / 1e6:.2f} MB "
        f"({num_elements:,} elements, {dtype}, {device})"
    )
    
    # Map device types to PyTorch device strings
    if device == "hpu":
        device = "hpu"  # Use HPU device for Intel Gaudi
    
    kv_caches = []
    for i in range(args.num_layers):
        kv_caches.append([torch.randn(num_elements, dtype=dtype, device=device) for _ in range(2)])
    return kv_caches


def create_xfer_descs(agent: NixlAgent, base_addr: int, num_blocks: int, block_len: int, mem_type: str):
    """Create transfer descriptors for a block range."""
    blocks_data = [(base_addr + i * block_len, block_len, 0) for i in range(num_blocks)]
    return agent.get_xfer_descs(blocks_data, mem_type)


def read_blocks(block_ids: Iterator[int], agent: NixlAgent,
                local_xfer_handle: str, remote_xfer_handle: str, sender_meta: NixlAgentMetadata, xfer_stats: NixlKVConnectorStats,):
    """ Read blocks from the sender's KV cache using NIXL. """
    if not block_ids:
        logging.warning("No block IDs provided for transfer.")
        return 0, 0
    
    try:
        local_ids = get_block_desc_ids(sender_meta.num_blocks, block_ids)
        remote_ids = get_block_desc_ids(sender_meta.num_blocks, block_ids)
        print(f"Reading blocks: {block_ids[0]} to {block_ids[-1]} (total {len(block_ids)})")

        xfer_handle = agent.make_prepped_xfer(
            "READ",
            local_xfer_handle,
            local_ids,
            remote_xfer_handle,
            remote_ids,
        )
        agent.transfer(xfer_handle)

        while agent.check_xfer_state(xfer_handle) != "DONE":
            time.sleep(0.00001)
        
        res = agent.get_xfer_telemetry(xfer_handle)
        print("res.descCount:", res.descCount)
        xfer_stats.record_transfer(res)
        agent.release_xfer_handle(xfer_handle)
        
        return
    except Exception as e:
        logging.error(f"Transfer failed in read_blocks: {e}", exc_info=True)
        raise


def add_remote_agent(agent: NixlAgent, sender_meta: NixlAgentMetadata, args: argparse.Namespace): 
    remote_agent_name = agent.add_remote_agent(sender_meta.agent_metadata)
    if isinstance(remote_agent_name, bytes):
        remote_agent_name = remote_agent_name.decode('utf-8')
    
    # Establish connection to the remote agent for the specified backend
    agent.make_connection(remote_agent_name, [args.nixl_backend])
    
    remote_xfer_descs = create_xfer_descs(
        agent, sender_meta.kv_caches_base_addr[0], sender_meta.num_blocks,
        sender_meta.block_len, args.nixl_memory_type
    )
    remote_xfer_handle = agent.prep_xfer_dlist(remote_agent_name, remote_xfer_descs)
    return agent, remote_xfer_handle


# ==============================================================================
# Processes
# ==============================================================================

def sender_process(args: argparse.Namespace):
    """
    Sender process: Allocates memory, registers it with NIXL,
    and waits to serve metadata and a shutdown signal.
    """
    logging.info("Sender starting...")
    agent = None
    config = nixl_agent_config(backends=[args.nixl_backend])
    #os.environ['UCX_NET_DEVICES'] = f'mlx5_{args.sender_device_id // 2}:1'
    
    try:
        sender_agent_id = str(uuid.uuid4())
        try:
            agent = NixlAgent(sender_agent_id, config)
        except nixl._bindings.nixlBackendError as e:
            logging.error(f"Failed to create NixlAgent with backend '{args.nixl_backend}': {e}")
            logging.error("Check logs above for detailed error messages about unsupported providers or initialization failures.")
            logging.error("Sender process cannot continue. Exiting.")
            return

        dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
        block_len = (
            args.num_heads * args.head_size * args.block_size * dtype.itemsize
        )

        ####### Prepare the KV cache for the sender #####
        kv_caches = allocate_kv_cache(args, block_len, dtype, args.sender_device_id)
        caches_data = []
        kv_caches_base_addr = []
        for cache_or_caches in kv_caches:
            # Normalize to always be a list of caches
            for cache in cache_or_caches:
                print("Cache shape:", cache.shape)
                base_addr = cache.data_ptr()
                region_len = args.num_blocks * block_len
                caches_data.append((base_addr, region_len, args.sender_device_id, ""))
                kv_caches_base_addr.append(base_addr)
        

        reg_descs = agent.get_reg_descs(caches_data, args.nixl_memory_type)
        print("================reg_descs.descCount:", reg_descs.descCount())
        #for i in range(reg_descs.descCount()):
        #    print(i, reg_descs[i])
        agent.register_memory(reg_descs, backends=[args.nixl_backend])
        blocks_data = []
        for base_addr in kv_caches_base_addr:
            for block_id in range(args.num_blocks):
                block_offset = block_id * block_len
                # (addr, len, device id)
                blocks_data.append(
                    (base_addr + block_offset, block_len, args.sender_device_id))

        # Register with NIXL.
        local_xfer_descs = agent.get_xfer_descs(blocks_data, args.nixl_memory_type)
        # for i in range(local_xfer_descs.descCount()):
        #     print(i, local_xfer_descs[i])
        agent.prep_xfer_dlist('NIXL_INIT_AGENT', local_xfer_descs)
        ##################################################

        metadata = NixlAgentMetadata(
            engine_id=f"sender-engine-{os.getpid()}",
            agent_metadata=agent.get_agent_metadata(),
            kv_caches_base_addr=kv_caches_base_addr,
            num_blocks=args.num_blocks,
            block_len=block_len,
        )
        encoder = msgspec.msgpack.Encoder()
        encoded_metadata = encoder.encode(metadata)
        with zmq.Context() as ctx, ctx.socket(zmq.ROUTER) as sock:
            zmq_addr = f"tcp://{ZMQ_HOST}:{ZMQ_BASE_PORT}"
            sock.bind(zmq_addr)
            logging.info(f"Sender listening for handshakes on {zmq_addr}")

            identity, _, msg = sock.recv_multipart()
            if msg == GET_META_MSG:
                sock.send_multipart((identity, b"", encoded_metadata))
                logging.info("Sent metadata to receiver.")
            else:
                raise RuntimeError(f"Expected metadata request, got: {msg}")

            identity, _, msg = sock.recv_multipart()
            if msg == SHUTDOWN_MSG:
                sock.send_multipart((identity, b"ack"))
                logging.info("Received shutdown signal. Sender will now exit.")
            else:
                logging.warning(f"Expected shutdown signal, got: {msg}")

    except Exception:
        logging.error("Sender process failed", exc_info=True)
    finally:
        if agent:
            try:
                # Attempt to clean up agent resources
                logging.info("Cleaning up sender agent resources...")
                del agent
            except Exception as e:
                logging.warning(f"Error cleaning up agent: {e}")
        logging.info("Sender shutting down.")

# ------------------------------------------------------------------------------

def send_shutdown_signal():
    """Helper function to send shutdown signal to sender."""
    try:
        with zmq.Context() as ctx, ctx.socket(zmq.REQ) as sock:
            zmq_addr = f"tcp://{ZMQ_HOST}:{ZMQ_BASE_PORT}"
            sock.connect(zmq_addr)
            logging.info("Sending shutdown signal to sender.")
            sock.send(SHUTDOWN_MSG)
            sock.recv()
    except zmq.ZMQError as e:
        logging.warning(f"Could not send shutdown signal to sender: {e}")


def receiver_process(args: argparse.Namespace):
    """
    Receiver process: Connects to sender, gets metadata, performs transfers,
    and reports benchmark results.
    """
    logging.info("Receiver starting...")
    agent = None
    config = nixl_agent_config(backends=[args.nixl_backend])
    #os.environ['UCX_NET_DEVICES'] = f'mlx5_{args.receiver_device_id // 2}:1'
    
    def cleanup_and_shutdown():
        """Send shutdown signal and cleanup."""
        send_shutdown_signal()
        logging.info("Receiver shutting down.")
    
    try:
        time.sleep(5)
        logging.info("Creating receiver agent...")
        receiver_agent_id = str(uuid.uuid4())
        try:
            agent = NixlAgent(receiver_agent_id, config)
        except nixl._bindings.nixlBackendError as e:
            logging.error(f"Failed to create NixlAgent with backend '{args.nixl_backend}': {e}")
            logging.error("Check logs above for detailed error messages about unsupported providers or initialization failures.")
            logging.error("Receiver process cannot continue. Exiting.")
            return
        logging.info(f"Created receiver agent {receiver_agent_id}")

        logging.info("Requesting metadata from sender...")
        with zmq.Context() as ctx, ctx.socket(zmq.REQ) as sock:
            zmq_addr = f"tcp://{ZMQ_HOST}:{ZMQ_BASE_PORT}"
            sock.connect(zmq_addr)
            logging.info(f"Requesting metadata from sender at {zmq_addr}...")
            sock.send(GET_META_MSG)
            metadata_bytes = sock.recv()

        sender_meta: NixlAgentMetadata = msgspec.msgpack.Decoder(NixlAgentMetadata).decode(metadata_bytes)
        logging.info(f"Received metadata from sender engine: {sender_meta.engine_id}")

        logging.info("Adding remote agent...")
        # agent, remote_xfer_handle = add_remote_agent(agent, sender_meta, args)
        remote_agent_name = agent.add_remote_agent(sender_meta.agent_metadata)
        # if isinstance(remote_agent_name, bytes):
        #     remote_agent_name = remote_agent_name.decode('utf-8')
        
        # Establish connection to the remote agent for the specified backend
        agent.make_connection(remote_agent_name, [args.nixl_backend])
        
        remote_blocks_data = []
        #print("sender_meta.kv_caches_base_addr:", sender_meta.kv_caches_base_addr)
        for base_addr in sender_meta.kv_caches_base_addr:
            for block_id in range(sender_meta.num_blocks):
                block_offset = block_id * sender_meta.block_len
                # (addr, len, device id)
                remote_blocks_data.append(
                    (base_addr + block_offset, sender_meta.block_len, args.sender_device_id))

        # Register with NIXL.
        remote_xfer_descs = agent.get_xfer_descs(remote_blocks_data, args.nixl_memory_type)
        remote_xfer_handle = agent.prep_xfer_dlist(remote_agent_name, remote_xfer_descs)
        logging.info("Remote agent added successfully")
        
        # Give sender time to be fully ready
        time.sleep(1)
        
        ####### Prepare the KV cache for the receiver #####
        logging.info("Preparing local KV cache...")
        dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
        block_len = sender_meta.block_len
        kv_caches = allocate_kv_cache(args, block_len, dtype, args.receiver_device_id)
        caches_data = []
        kv_caches_base_addr = []
        for cache_or_caches in kv_caches:
            # Normalize to always be a list of caches
            for cache in cache_or_caches:
                base_addr = cache.data_ptr()
                region_len = args.num_blocks * block_len
                caches_data.append((base_addr, region_len, args.receiver_device_id, ""))
                kv_caches_base_addr.append(base_addr)
        

        reg_descs = agent.get_reg_descs(caches_data, args.nixl_memory_type)
        agent.register_memory(reg_descs, backends=[args.nixl_backend])

        blocks_data = []
        for base_addr in kv_caches_base_addr:
            for block_id in range(args.num_blocks):
                block_offset = block_id * block_len
                # (addr, len, device id)
                blocks_data.append(
                    (base_addr + block_offset, block_len, args.receiver_device_id))

        # Register with NIXL.
        local_xfer_descs = agent.get_xfer_descs(blocks_data, args.nixl_memory_type)
        local_xfer_handle = agent.prep_xfer_dlist('NIXL_INIT_AGENT', local_xfer_descs)

        logging.info("Local setup complete")
        #################################################

        logging.info(f"Starting transfer loop for {args.num_iterations} iterations...")
        xfer_stats = NixlKVConnectorStats()

        # input_tokens / block_size -> block_ids
        # block_ids * num_layers * 2
        block_ids = list(range(args.input_tokens // args.block_size * args.num_layers * 2))
        for i in range(args.num_iterations):
            if i % 10 == 0:  # Log every 10th iteration
                logging.info(f"Transfer iteration {i+1}/{args.num_iterations}")
            # do transfer
            read_blocks(block_ids, agent, local_xfer_handle, remote_xfer_handle, sender_meta, xfer_stats)

        # Print summary after successful completion
        logging.info("All transfers completed successfully")
        logging.info(xfer_stats.reduce())

        # Send shutdown signal only after successful completion
        logging.info("Sending shutdown signal after successful completion")
        cleanup_and_shutdown()
        return  # Exit receiver process after successful completion

    except Exception as e:
        logging.error(f"Receiver process failed with exception: {e}", exc_info=True)
        # Send shutdown signal even on error to clean up sender
        logging.info("Sending shutdown signal due to error")
        cleanup_and_shutdown()
        return  # Exit receiver process after error
    finally:
        if agent:
            try:
                # Attempt to clean up agent resources
                logging.info("Cleaning up receiver agent resources...")
                # Explicitly clean up any backend connections
                if hasattr(agent, 'cleanup'):
                    agent.cleanup()
                del agent
                agent = None
                logging.info("Receiver agent cleanup completed")
            except Exception as e:
                logging.warning(f"Error cleaning up receiver agent: {e}")
        logging.info("Receiver process function finished")


# ==============================================================================
# Main
# ==============================================================================
"""
kv_cache_size = num_blocks * block_len
              = num_blocks * (num_heads * head_size * 2 * block_size * dtype.itemsize)
              = num_layers * tokens_num / block_size * block_len
num_blocks = num_layers * tokens_num / block_size
"""
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description="Benchmark script for nixl._api performance.")
    parser.add_argument("--num-blocks", type=int, default=1000)
    parser.add_argument("--block-size", type=int, default=16, help="Tokens per block")
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--input-tokens", type=int, default=256)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--nixl-memory-type", type=str, default="VRAM", choices=["DRAM", "VRAM"])
    parser.add_argument("--device-type", type=str, default="cpu", choices=["cpu", "cuda", "hpu", "xpu"])
    parser.add_argument("--nixl_backend", type=str, default="OFI")
    parser.add_argument("--provider", type=str, default=None, help="OFI provider (e.g., tcp, shm, verbs)")
    parser.add_argument("--ucx-transport", type=str, default=None, help="default is tcp, you might configure as 'cuda_copy,sm'")
    parser.add_argument("--debug-ucx", action="store_true",
                        help="Enable debug mode for UCX backend (if using UCX backend)")
    parser.add_argument("--sender-device-id", type=int, default=0, help="Sender device ID to use if applicable")
    parser.add_argument("--receiver-device-id", type=int, default=0, help="Receiver device ID to use if applicable")
    args = parser.parse_args()
    
    # NIXL_PLUGIN_DIR=/workspace/nixl/nixl-nixl_libfabric/build/cp310/src/plugins/libfabric python nixl_api.py  --device-type hpu --nixl_backend libfabric
    
    if args.debug_ucx:
        os.environ['UCX_PROTO_INFO'] = 'y'
    if args.ucx_transport:
        os.environ['UCX_TLS'] = args.ucx_transport
    if args.provider:
        # Set libfabric provider environment variable
        os.environ['FI_PROVIDER'] = args.provider 
    if args.device_type == "cpu":
        args.nixl_memory_type = "DRAM"
    elif args.device_type == "cuda":
        args.nixl_memory_type = "VRAM"
    elif args.device_type == "hpu":
        args.nixl_memory_type = "VRAM"
    elif args.device_type == "xpu":
        args.nixl_memory_type = "VRAM"
    #args.dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    if args.device_type == "cuda" and not torch.cuda.is_available():
        logging.error("CUDA device specified but not available. Exiting.")
        exit(1)
    elif args.device_type == "xpu" and not torch.xpu.is_available():
        logging.error("XPU device specified but not available. Exiting.")
        exit(1)
    elif args.device_type == "hpu":
        try:
            import habana_frameworks.torch.core as htcore
            # Check if HPU is available
            if not hasattr(htcore, 'hpu') or not htcore.is_available():
                logging.error("HPU device specified but not available. Exiting.")
                exit(1)
        except ImportError:
            logging.error("HPU device specified but habana_frameworks not installed. Exiting.")
            exit(1)

    sender = multiprocessing.Process(target=sender_process, args=(args,), name="Sender")
    receiver = multiprocessing.Process(target=receiver_process, args=(args,), name="Receiver")

    try:
        sender.start()
        receiver.start()
        receiver.join(timeout=100)  # 10 second timeout
        
        if receiver.is_alive():
            logging.warning("Receiver process didn't exit cleanly, terminating...")
            receiver.terminate()
            receiver.join(timeout=5)
        
        # Wait for sender to shutdown gracefully
        if sender.is_alive():
            logging.info("Waiting for sender to shutdown...")
            sender.join(timeout=10)
            if sender.is_alive():
                logging.warning("Sender didn't shutdown gracefully, terminating...")
                sender.terminate()
                sender.join(timeout=2)
        
    except KeyboardInterrupt:
        logging.info("Interrupted by user, terminating processes...")
    finally:
        for proc in (sender, receiver):
            if proc.is_alive():
                logging.info(f"Force terminating {proc.name}...")
                proc.terminate()
                proc.join(timeout=2)
                if proc.is_alive():
                    logging.warning(f"Failed to terminate {proc.name}")
        logging.info("Benchmark finished.")
