import subprocess
import os
import sys
import time
import argparse
import logging
from sglang.utils import wait_for_server

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_gpu_count():
    """获取可用的GPU数量"""
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
    """获取GPU内存信息"""
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
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Launch 30B model server with multi-GPU support")
    parser.add_argument("--model-path", 
                       default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                       help="Path to the model")
    parser.add_argument("--port", type=int, default=34882, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--mem-fraction-static", type=float, default=0.8, 
                       help="Memory fraction to use per GPU")
    parser.add_argument("--gpus", type=str, default=None,
                       help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')")
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                       help="Tensor parallel size (number of GPUs for model parallelism)")
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
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
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
    
    return parser.parse_args()

def build_launch_command(args):
    """构建启动命令"""
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", args.model_path,
        "--host", args.host,
        "--port", str(args.port),
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--dtype", args.dtype
    ]
    
    
    if args.tensor_parallel_size:
        cmd.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])
        logger.info(f"Using tensor parallel size: {args.tensor_parallel_size}")
    
    # 添加可选参数
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
    
    return cmd

def check_system_requirements():
    """检查系统要求"""
    logger.info("Checking system requirements...")
    
    # 检查GPU
    gpu_count = get_gpu_count()
    if gpu_count == 0:
        logger.warning("No CUDA GPUs detected. The model will run on CPU, which may be very slow.")
    else:
        logger.info(f"Detected {gpu_count} GPU(s)")
        gpu_info = get_gpu_memory_info()
        for info in gpu_info:
            logger.info(info)
    
    # 检查模型路径
    model_path = "/root/Qwen/Qwen3-30B-A3B-Instruct-2507"
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        return False
    
    # 检查端口
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 34882))
    sock.close()
    if result == 0:
        logger.warning("Port 34882 is already in use")
    
    return True

def main():
    """主函数"""
    logger.info("Starting 30B model server with multi-GPU support...")
    
    # 检查系统要求
    if not check_system_requirements():
        logger.error("System requirements check failed")
        sys.exit(1)
    
    # 解析参数
    args = parse_arguments()
    
    # 自动检测GPU数量并设置tensor parallel size
    gpu_count = get_gpu_count()
    if args.tensor_parallel_size is None and gpu_count > 1:
        # 对于30B模型，建议使用2-4个GPU进行tensor parallel
        if gpu_count >= 4:
            args.tensor_parallel_size = 4
        elif gpu_count >= 2:
            args.tensor_parallel_size = 2
        else:
            args.tensor_parallel_size = 1
        logger.info(f"Auto-detected tensor parallel size: {args.tensor_parallel_size}")
    
    # 如果没有指定GPU，使用所有可用GPU
    if args.gpus is None and gpu_count > 0:
        args.gpus = ",".join(map(str, range(gpu_count)))
        logger.info(f"Auto-detected GPUs: {args.gpus}")
    
    # 构建启动命令
    cmd = build_launch_command(args)
    
    logger.info("Launch command:")
    logger.info(" ".join(cmd))
    
    # 启动服务器
    try:
        logger.info("Starting server...")
        server_process = subprocess.Popen(cmd)
        
        # 等待服务器启动
        logger.info("Waiting for server to start...")
        wait_for_server(f"http://localhost:{args.port}")
        
        logger.info(f"Server started successfully on http://localhost:{args.port}")
        logger.info(f"Process ID: {server_process.pid}")
        
        # 保持脚本运行
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
