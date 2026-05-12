import subprocess
import os
import sys
import time
import argparse
import logging
from sglang.utils import wait_for_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_gpu_count():
    """Get the number of available GPUs"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            return 0
    except ImportError:
        logger.warning("PyTorch not available, cannot detect GPU count")
        return 0

def get_gpu_memory_info():
    """Get GPU memory information"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                gpu_info.append(f"GPU{i}: {memory:.1f}GB")
            return gpu_info
        else:
            return []
    except ImportError:
        return []

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Launch 30B model server with multi-GPU support")
    parser.add_argument("--model-path", 
                       default=None,
                       help="Path to the model")
    parser.add_argument("--port", type=int, default=34882, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--mem-fraction-static", type=float, default=0.8, 
                       help="Memory fraction to use per GPU")
    parser.add_argument("--gpus", type=str, default=None,
                       help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')")
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                       help="Tensor parallel size (number of GPUs for model parallelism)")
    parser.add_argument("--expert-parallel-size", type=int, default=None,
                       help="Expert parallel size for MoE models")
    parser.add_argument("--data-parallel-size", type=int, default=None,
                       help="Data parallel size")
    parser.add_argument("--dist-init-addr", type=str, default=None,
                       help="Distributed init address, e.g. 10.0.0.1:5000")
    parser.add_argument("--nnodes", type=int, default=None,
                       help="Number of nodes for multi-node serving")
    parser.add_argument("--node-rank", type=int, default=None,
                       help="Current node rank for multi-node serving")
    parser.add_argument("--enable-dp-attention", action="store_true",
                       help="Enable data parallel attention")
    parser.add_argument("--kv-cache-dtype", default=None,
                       help="KV cache dtype (e.g., fp8_e4m3)")
    parser.add_argument("--max-concurrent-requests", type=int, default=100,
                       help="Maximum concurrent requests")
    parser.add_argument("--max-model-len", type=int, default=8192,
                       help="Maximum model length")
    parser.add_argument("--trust-remote-code", action="store_true",
                       help="Trust remote code")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"],
                       help="Model data type")
    parser.add_argument("--disable-log-requests", action="store_true",
                       help="Disable request logging")
    parser.add_argument("--disable-log-stats", action="store_true",
                       help="Disable statistics logging")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                       help="GPU memory utilization ratio")
    parser.add_argument("--swap-space", type=int, default=4,
                       help="CPU swap space size (GiB)")
    parser.add_argument("--max-paddings", type=int, default=256,
                       help="Maximum number of paddings in a batch")
    parser.add_argument("--max-seqs", type=int, default=256,
                       help="Maximum number of sequences per GPU")
    parser.add_argument("--quantization", default=None, choices=["awq", "gptq", "squeezellm"],
                       help="Quantization method")
    parser.add_argument("--enforce-eager", action="store_true",
                       help="Enforce eager mode")
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192,
                       help="Maximum number of batched tokens")
    parser.add_argument("--max-num-seqs", type=int, default=256,
                       help="Maximum number of sequences")
    parser.add_argument("--max-running-requests", type=int, default=200,
                       help="Maximum number of running requests")
    parser.add_argument("--pipeline-parallel-size", type=int, default=None,
                       help="Pipeline parallel size")
    parser.add_argument("--tool-call-parser", type=str, default=None,
                       help="Tool call parser")
    parser.add_argument("--reasoning-parser", type=str, default=None,
                       help="Reasoning parser")
    parser.add_argument("--context-length", type=int, default=None,
                       help="Context length")
    parser.add_argument("--page-size", type=int, default=None,
                       help="Page size")
    parser.add_argument("--speculative-algorithm", type=str, default=None,
                       help="Speculative algorithm")
    parser.add_argument("--speculative-num-steps", type=int, default=None,
                       help="Speculative number of steps")
    parser.add_argument("--speculative-eagle-topk", type=int, default=None,
                       help="Speculative eagle topk")
    parser.add_argument("--speculative-num-draft-tokens", type=int, default=None,
                       help="Speculative number of draft tokens")
    parser.add_argument("--chat-template", type=str, default=None,
                       help="Chat template")
    return parser.parse_args()

def build_launch_command(args):
    """Build launch command"""
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", args.model_path,
        "--host", args.host,
        "--port", str(args.port),
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--dtype", args.dtype,
        "--max-running-requests", str(args.max_running_requests),
    ]

    if args.chat_template:
        cmd.extend(["--chat-template", args.chat_template])
        logger.info(f"Using chat template: {args.chat_template}")
    
    if args.pipeline_parallel_size:
        cmd.extend(["--pipeline-parallel-size", str(args.pipeline_parallel_size)])
        logger.info(f"Using pipeline parallel size: {args.pipeline_parallel_size}")

    if args.tensor_parallel_size:
        cmd.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])
        logger.info(f"Using tensor parallel size: {args.tensor_parallel_size}")

    if args.expert_parallel_size:
        cmd.extend(["--expert-parallel-size", str(args.expert_parallel_size)])
        logger.info(f"Using expert parallel size: {args.expert_parallel_size}")

    if args.data_parallel_size:
        # Translate user-facing argument to sglang's short-form CLI flag.
        cmd.extend(["--data-parallel-size", str(args.data_parallel_size)])
        logger.info(f"Using data parallel size: {args.data_parallel_size}")

    if args.dist_init_addr:
        cmd.extend(["--dist-init-addr", args.dist_init_addr])
        logger.info(f"Using distributed init address: {args.dist_init_addr}")

    if args.nnodes:
        cmd.extend(["--nnodes", str(args.nnodes)])
        logger.info(f"Using nnodes: {args.nnodes}")

    if args.node_rank is not None:
        cmd.extend(["--node-rank", str(args.node_rank)])
        logger.info(f"Using node rank: {args.node_rank}")

    if args.enable_dp_attention:
        cmd.append("--enable-dp-attention")
        logger.info("Enabled data parallel attention")

    if args.kv_cache_dtype:
        cmd.extend(["--kv-cache-dtype", args.kv_cache_dtype])
        logger.info(f"Using kv cache dtype: {args.kv_cache_dtype}")
    
    # Add optional parameters
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    
    if args.disable_log_requests:
        cmd.append("--disable-log-requests")
    
    if args.disable_log_stats:
        cmd.append("--disable-log-stats")
    
    if args.quantization:
        cmd.extend(["--quantization", args.quantization])
        logger.info(f"Using quantization: {args.quantization}")
    
    if args.enforce_eager:
        cmd.append("--enforce-eager")

    if args.tool_call_parser:
        cmd.extend(["--tool-call-parser", args.tool_call_parser])
        logger.info(f"Using tool call parser: {args.tool_call_parser}")

    if args.reasoning_parser:
        cmd.extend(["--reasoning-parser", args.reasoning_parser])
        logger.info(f"Using reasoning parser: {args.reasoning_parser}")

    if args.context_length:
        cmd.extend(["--context-length", str(args.context_length)])
        logger.info(f"Using context length: {args.context_length}")

    if args.page_size:
        cmd.extend(["--page-size", str(args.page_size)])
        logger.info(f"Using page size: {args.page_size}")
    
    if args.speculative_algorithm:
        cmd.extend(["--speculative-algorithm", args.speculative_algorithm])
        logger.info(f"Using speculative algorithm: {args.speculative_algorithm}")
    
    if args.speculative_num_steps:
        cmd.extend(["--speculative-num-steps", str(args.speculative_num_steps)])
        logger.info(f"Using speculative number of steps: {args.speculative_num_steps}")
    
    if args.speculative_eagle_topk:
        cmd.extend(["--speculative-eagle-topk", str(args.speculative_eagle_topk)])
        logger.info(f"Using speculative eagle topk: {args.speculative_eagle_topk}")
    
    if args.speculative_num_draft_tokens:
        cmd.extend(["--speculative-num-draft-tokens", str(args.speculative_num_draft_tokens)])
        logger.info(f"Using speculative number of draft tokens: {args.speculative_num_draft_tokens}")

    return cmd

def check_system_requirements(model_path: str, port: int):
    """Check system requirements"""
    logger.info("Checking system requirements...")
    
    # Check GPU
    gpu_count = get_gpu_count()
    if gpu_count == 0:
        logger.warning("No CUDA GPUs detected. The model will run on CPU, which may be very slow.")
    else:
        logger.info(f"Detected {gpu_count} GPU(s)")
        gpu_info = get_gpu_memory_info()
        for info in gpu_info:
            logger.info(info)
    
    # Check model path
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        return False
    
    # Check port
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    if result == 0:
        logger.warning(f"Port {port} is already in use")
    
    return True

def main():
    """Main function"""
    logger.info("Starting 30B model server with multi-GPU support...")

    # Parse arguments
    args = parse_arguments()

    # Check system requirements (using the actual model path and port)
    if not check_system_requirements(args.model_path, args.port):
        logger.error("System requirements check failed")
        sys.exit(1)
    
    # Automatically detect GPU count and set tensor parallel size
    gpu_count = get_gpu_count()
    if args.tensor_parallel_size is None and gpu_count > 1:
        # For 30B model, it is recommended to use 2-4 GPUs for tensor parallel
        if gpu_count >= 4:
            args.tensor_parallel_size = 4
        elif gpu_count >= 2:
            args.tensor_parallel_size = 2
        else:
            args.tensor_parallel_size = 1
        logger.info(f"Auto-detected tensor parallel size: {args.tensor_parallel_size}")
    
    # If no GPU is specified, use all available GPUs
    if args.gpus is None and gpu_count > 0:
        args.gpus = ",".join(map(str, range(gpu_count)))
        logger.info(f"Auto-detected GPUs: {args.gpus}")
    
    # Build launch command
    cmd = build_launch_command(args)
    
    logger.info("Launch command:")
    logger.info(" ".join(cmd))

    # Start server
    try:
        logger.info("Starting server...")
        server_process = subprocess.Popen(cmd)
        
        # Wait for server to start
        logger.info("Waiting for server to start...")
        wait_for_server(f"http://localhost:{args.port}")
        
        logger.info(f"Server started successfully on http://localhost:{args.port}")
        logger.info(f"Process ID: {server_process.pid}")
        
        # Keep script running
        try:
            server_process.wait()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
            server_process.terminate()
            server_process.wait()
            logger.info("Server stopped")
            
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
