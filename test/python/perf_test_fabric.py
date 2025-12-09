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
# ==============================================================================
# Configuration
# ==============================================================================

ZMQ_HOST = "127.0.0.1"
# ZMQ_BASE_PORT = 15555
GET_META_MSG = b"get_meta_msg"
SHUTDOWN_MSG = b"shutdown_msg"
RECEIVER_META_MSG = b"receiver_meta"  # For symmetric connection pattern

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


# ==============================================================================
# Helpers
# ==============================================================================
import hashlib
def tensor_hash(tensor: torch.Tensor) -> int:
    """Calculate the hash value of the tensor."""
    tensor_bytes = tensor.clone().detach().cpu().numpy().tobytes()
    hash_object = hashlib.blake2b(tensor_bytes)
    hash_hex = hash_object.hexdigest()
    return int(hash_hex[:16], 16)

def get_block_desc_ids(num_total_blocks: int, block_ids: Iterator[int]) -> List[int]:
    """
    Maps logical block IDs to the indices of their transfer descriptors.
    """
    return list(block_ids)

def allocate_kv_cache(args, device_id):
    total_bytes = args.MB_size * 1024 * 1024
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    bytes_per_element = dtype.itemsize

    num_elements = total_bytes // bytes_per_element

    device_str = args.device_type
    if args.device_type != "cpu":
        device_str = f"{args.device_type}:{device_id}"
    kv_cache = torch.randn(num_elements, dtype=dtype, device=device_str)
    print(f"KV cache allocated on {device_str} with shape {kv_cache.shape} and dtype {kv_cache.dtype}")
    return kv_cache


def create_xfer_descs(agent: NixlAgent, base_addr: int, num_blocks: int, block_len: int, mem_type: str, device_id: int):
    """Create transfer descriptors for a block range."""
    blocks_data = [(base_addr + i * block_len, block_len, device_id) for i in range(num_blocks)]
    return agent.get_xfer_descs(blocks_data, mem_type)


def read_blocks(block_ids: Iterator[int], agent: NixlAgent,
                local_xfer_handle: str, remote_xfer_handle: str, sender_meta: NixlAgentMetadata):
    """ Read blocks from the sender's KV cache using NIXL. """
    if not block_ids:
        logging.warning("No block IDs provided for transfer.")
        return 0, 0

    local_ids = get_block_desc_ids(sender_meta.num_blocks, block_ids)
    remote_ids = get_block_desc_ids(sender_meta.num_blocks, block_ids)


    xfer_handle = agent.make_prepped_xfer(
        "READ",
        local_xfer_handle,
        local_ids,
        remote_xfer_handle,
        remote_ids,
    )
    t0 = time.perf_counter_ns()
    agent.transfer(xfer_handle)

    while agent.check_xfer_state(xfer_handle) != "DONE":
        time.sleep(0.00001)

    t1 = time.perf_counter_ns()  # End timing after verification
    agent.release_xfer_handle(xfer_handle)

    return (t1 - t0) / 1e6, len(local_ids) * sender_meta.block_len  # Return latency in ms


def summary(latencies: List[float], sender_meta: NixlAgentMetadata, args: argparse.Namespace, total_data_transferred: int, agent: NixlAgent):
    total_time = sum(latencies)
    avg_latency_ms = (total_time / args.num_iterations) if args.num_iterations > 0 else 0
    # total_data_transferred (bytes) / total_time (ms) * 1000 (ms/s) / 1e9 (bytes/GB) = GB/s
    throughput_gbps = (total_data_transferred / total_time) * 1e-6 if total_time > 0 else 0

    # Extract backend name from the dictionary for cleaner display
    backend_name = list(agent.backends.keys())[0] if agent.backends else "Unknown"

    print("\n" + "=" * 60)
    print(" NIXL API Performance Benchmark Results")
    print("=" * 60)
    print(f" Hardware Memory Type:  {args.nixl_memory_type}")
    print(f" Device Type:           {args.device_type}")
    print(f" NIXL Backend:          {backend_name}")
    print(f" Total Iterations:      {args.num_iterations:,}")
    print(f" Blocks per Transfer:   {args.blocks_per_xfer:,}")
    print(f" Data per Transfer:     {args.MB_size} MB")
    print("-" * 60)
    print(f" Average Latency:       {avg_latency_ms:.3f} ms")
    print(f" Total Throughput:      {throughput_gbps:.3f} GB/s")
    print(f" Total Data Transferred:      {total_data_transferred / 1024 / 1024 / 1024:.3f} GB")
    print("-" * 60)

    # Format latency statistics nicely
    stats = pd.Series(latencies).describe()
    print(" Latency Statistics (ms):")
    print(f"   Count:               {stats['count']:.0f}")
    print(f"   Mean:                {stats['mean']:.3f}")
    print(f"   Std Dev:             {stats['std']:.3f}")
    print(f"   Min:                 {stats['min']:.3f}")
    print(f"   25th %ile:           {stats['25%']:.3f}")
    print(f"   Median:              {stats['50%']:.3f}")
    print(f"   75th %ile:           {stats['75%']:.3f}")
    print(f"   Max:                 {stats['max']:.3f}")
    print("=" * 60 + "\n")


def add_remote_agent(agent: NixlAgent, sender_meta: NixlAgentMetadata, args: argparse.Namespace, device_id=0):
    remote_agent_name = agent.add_remote_agent(sender_meta.agent_metadata)
    if isinstance(remote_agent_name, bytes):
        remote_agent_name = remote_agent_name.decode('utf-8')

    # Establish connection to the remote agent for the specified backend
    agent.make_connection(remote_agent_name, [args.nixl_backend])

    remote_xfer_descs = create_xfer_descs(
        agent, sender_meta.kv_caches_base_addr[0], sender_meta.num_blocks,
        sender_meta.block_len, args.nixl_memory_type, device_id
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

        ####### Prepare the KV cache for the sender #####
        # device_id = 0
        kv_cache = allocate_kv_cache(args, args.sender_device_id)
        block_len = args.MB_size * 1024 * 1024
        logging.info("Allocated sender KV cache")
        logging.info(f"Tensor hash: {tensor_hash(kv_cache)}")
        # torch.save(kv_cache.cpu(), "sent_kv_cache.pt")
        reg_descs = agent.get_reg_descs(
            [(kv_cache.data_ptr(), kv_cache.numel() * kv_cache.element_size(), args.sender_device_id, "")],
            args.nixl_memory_type
        )
        agent.register_memory(reg_descs, backends=[args.nixl_backend])

        base_addr = kv_cache.data_ptr()
        local_xfer_descs = create_xfer_descs(agent, base_addr, args.num_blocks, block_len, args.nixl_memory_type, args.sender_device_id)
        agent.prep_xfer_dlist('NIXL_INIT_AGENT', local_xfer_descs)
        ##################################################

        metadata = NixlAgentMetadata(
            engine_id=f"sender-engine-{os.getpid()}",
            agent_metadata=agent.get_agent_metadata(),
            kv_caches_base_addr=[base_addr],
            num_blocks=args.num_blocks,
            block_len=block_len,
        )
        encoder = msgspec.msgpack.Encoder()
        encoded_metadata = encoder.encode(metadata)

        with zmq.Context() as ctx, ctx.socket(zmq.ROUTER) as sock:
            zmq_addr = f"tcp://{ZMQ_HOST}:{args.zmq_port}"
            sock.bind(zmq_addr)
            logging.info(f"Sender listening for handshakes on {zmq_addr}")

            # Step 1: Receiver requests sender's metadata
            identity, _, msg = sock.recv_multipart()
            if msg == GET_META_MSG:
                sock.send_multipart((identity, b"", encoded_metadata))
                logging.info("Sent metadata to receiver.")
            else:
                raise RuntimeError(f"Expected metadata request, got: {msg}")

            # Step 2: Receiver sends its metadata back (symmetric connection pattern)
            identity, _, msg = sock.recv_multipart()
            if msg.startswith(RECEIVER_META_MSG):
                # Extract receiver metadata from message
                receiver_metadata_bytes = msg[len(RECEIVER_META_MSG):]
                receiver_meta = msgspec.msgpack.Decoder(NixlAgentMetadata).decode(receiver_metadata_bytes)
                logging.info(f"Received metadata from receiver engine: {receiver_meta.engine_id}")

                # Add remote agent and establish connection (symmetric with receiver)
                logging.info("Adding receiver as remote agent...")
                remote_agent_name = agent.add_remote_agent(receiver_meta.agent_metadata)
                if isinstance(remote_agent_name, bytes):
                    remote_agent_name = remote_agent_name.decode('utf-8')
                agent.make_connection(remote_agent_name, [args.nixl_backend])
                logging.info(f"Sender connected to receiver: {remote_agent_name}")

                # Send ACK to receiver
                sock.send_multipart((identity, b"", b"connected"))
            else:
                raise RuntimeError(f"Expected receiver metadata, got: {msg}")

            # Step 3: Wait for shutdown signal
            identity, _, msg = sock.recv_multipart()
            if msg == SHUTDOWN_MSG:
                sock.send_multipart((identity, b"", b"ack"))
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

def send_shutdown_signal(args: argparse.Namespace):
    """Helper function to send shutdown signal to sender."""
    try:
        with zmq.Context() as ctx, ctx.socket(zmq.REQ) as sock:
            zmq_addr = f"tcp://{ZMQ_HOST}:{args.zmq_port}"
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
        send_shutdown_signal(args)
        logging.info("Receiver shutting down.")

    try:
        # Wait briefly to ensure sender has started
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

        ####### Prepare the KV cache for the receiver FIRST (before connection) #####
        logging.info("Preparing local KV cache...")
        local_kv_cache = allocate_kv_cache(args, args.receiver_device_id)
        local_base_addr = local_kv_cache.data_ptr()
        block_len = args.MB_size * 1024 * 1024
        logging.info("Registering local memory...")
        reg_descs = agent.get_reg_descs(
            [(local_base_addr, local_kv_cache.numel() * local_kv_cache.element_size(), args.receiver_device_id, "")],
            args.nixl_memory_type
        )
        agent.register_memory(reg_descs, backends=[args.nixl_backend])

        # Prepare receiver's metadata for symmetric connection
        receiver_metadata = NixlAgentMetadata(
            engine_id=f"receiver-engine-{os.getpid()}",
            agent_metadata=agent.get_agent_metadata(),
            kv_caches_base_addr=[local_base_addr],
            num_blocks=args.num_blocks,
            block_len=block_len,
        )
        encoder = msgspec.msgpack.Encoder()
        encoded_receiver_metadata = encoder.encode(receiver_metadata)
        #################################################

        logging.info("Requesting metadata from sender...")
        with zmq.Context() as ctx, ctx.socket(zmq.REQ) as sock:
            zmq_addr = f"tcp://{ZMQ_HOST}:{args.zmq_port}"
            sock.connect(zmq_addr)

            # Step 1: Request sender's metadata
            logging.info(f"Requesting metadata from sender at {zmq_addr}...")
            sock.send(GET_META_MSG)
            metadata_bytes = sock.recv()
            sender_meta = msgspec.msgpack.Decoder(NixlAgentMetadata).decode(metadata_bytes)
            logging.info(f"Received metadata from sender engine: {sender_meta.engine_id}")

            # Step 2: Add sender as remote agent
            logging.info("Adding sender as remote agent...")
            remote_agent_name = agent.add_remote_agent(sender_meta.agent_metadata)
            if isinstance(remote_agent_name, bytes):
                remote_agent_name = remote_agent_name.decode('utf-8')
            agent.make_connection(remote_agent_name, [args.nixl_backend])
            logging.info(f"Receiver connected to sender: {remote_agent_name}")

            # Step 3: Send receiver's metadata to sender for symmetric connection
            logging.info("Sending receiver metadata to sender for symmetric connection...")
            sock.send(RECEIVER_META_MSG + encoded_receiver_metadata)
            ack = sock.recv()
            if ack == b"connected":
                logging.info("Sender confirmed symmetric connection established")
            else:
                logging.warning(f"Unexpected response from sender: {ack}")

        # Prepare remote transfer descriptors
        remote_xfer_descs = create_xfer_descs(
            agent, sender_meta.kv_caches_base_addr[0], sender_meta.num_blocks,
            sender_meta.block_len, args.nixl_memory_type, args.sender_device_id
        )
        remote_xfer_handle = agent.prep_xfer_dlist(remote_agent_name, remote_xfer_descs)

        # Prepare local transfer descriptors (KV cache already allocated above)
        logging.info("Creating local transfer descriptors...")
        local_xfer_descs = create_xfer_descs(
            agent, local_base_addr, sender_meta.num_blocks, sender_meta.block_len, args.nixl_memory_type, args.receiver_device_id
        )
        local_xfer_handle = agent.prep_xfer_dlist('NIXL_INIT_AGENT', local_xfer_descs)
        logging.info("Local setup complete")

        latencies = []
        total_data_transferred = 0
        logging.info(f"Starting transfer loop for {args.num_iterations} iterations...")

        for i in range(args.num_iterations):
            # if i % 10 == 0:  # Log every 10th iteration
            logging.info(f"Transfer iteration {i+1}/{args.num_iterations}")
            start_idx = 0
            print(f"Transferring blocks {start_idx} to {start_idx + args.blocks_per_xfer - 1}")
            block_ids = list(range(start_idx, start_idx + args.blocks_per_xfer))
            # do transfer
            latency, _ = read_blocks(block_ids, agent, local_xfer_handle, remote_xfer_handle, sender_meta)
            latencies.append(latency)
            total_data_transferred += args.MB_size * 1024 * 1024

        logging.info("Transferred local KV cache")
        logging.info(f"Tensor hash: {tensor_hash(local_kv_cache)}")
        # torch.save(local_kv_cache.cpu(), "received_kv_cache.pt")
        # Print summary after successful completion
        logging.info("All transfers completed successfully")
        summary(latencies, sender_meta, args, total_data_transferred, agent)

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

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description="Benchmark script for nixl._api performance.")
    parser.add_argument("--num-blocks", type=int, default=1)
    # parser.add_argument("--block-size", type=int, default=16, help="Tokens per block")
    # parser.add_argument("--num-heads", type=int, default=32)
    # parser.add_argument("--head-size", type=int, default=128)
    # parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--blocks-per-xfer", type=int, default=1)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--MB-size", type=int, default=1, help="Size of KV cache in MB")
    parser.add_argument("--nixl-memory-type", type=str, default="VRAM", choices=["DRAM", "VRAM"])
    parser.add_argument("--device-type", type=str, default="cpu", choices=["cpu", "cuda", "xpu", "hpu"])
    parser.add_argument("--nixl_backend", type=str, default="OFI")
    parser.add_argument("--provider", type=str, default=None, help="OFI provider (e.g., tcp, shm, verbs)")
    parser.add_argument("--ucx-transport", type=str, default=None, help="default is tcp, you might configure as 'cuda_copy,sm'")
    parser.add_argument("--debug-ucx", action="store_true",
                        help="Enable debug mode for UCX backend (if using UCX backend)")
    parser.add_argument("--sender-device-id", type=int, default=0, help="Sender device ID to use if applicable")
    parser.add_argument("--receiver-device-id", type=int, default=0, help="Receiver device ID to use if applicable")
    parser.add_argument("--zmq-port", type=int, default=8110, help="Base port for ZMQ communication")
    args = parser.parse_args()
    assert args.num_blocks == 1

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
    elif args.device_type == "xpu":
        args.nixl_memory_type = "VRAM"
    elif args.device_type == "hpu":
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
        receiver.join(timeout=1000)  # 10 second timeout

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
