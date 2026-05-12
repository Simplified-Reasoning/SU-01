# reward_model_server.py
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Union, List, Optional
import asyncio
import signal
import sys
import gc
import time
import argparse
import os
from p1 import compute_score_p1, Model_args
from proof_verifier import compute_score_proof

app = FastAPI()

# 全局变量用于优雅关闭
shutdown_event = asyncio.Event()

# 全局变量存储模型端口
MODEL_PORT = int(os.environ.get('MODEL_PORT', 34882))

class RewardRequest(BaseModel):
    response: str
    label: Optional[Union[str, List[str]]] = None  # 可以是 None
    points: Optional[List[float]] = None
    question: Optional[str] = None
    use_xverify: bool = False
    is_proof: bool = False
    reviewer: Optional[str] = None
    reviews: Optional[int] = None
    marking: Optional[List[str]] = None  # 用于 marking 模式的评分标准列表

@app.post("/")
async def evaluate_reward(req: RewardRequest):
    # Log full payload to debug field parsing issues (e.g., missing is_proof flag).
    try:
        print("Received Request:", req.dict())
    except Exception:
        print("Received Request:", req)
    try:
        # 判断是否为 proof 请求：显式设置 is_proof，或没有 label 且没有 points，或有 marking
        is_proof_request = req.is_proof or (req.label is None and req.points is None) or (req.marking is not None)
        if is_proof_request:
            if not req.question:
                raise HTTPException(status_code=400, detail="`question` is required for proof verification")
            verifier_model = Model_args.model_name if hasattr(Model_args, "model_name") else "gpt-oss-120b"
            verifier_api_key = getattr(Model_args, "api_key", None)
            if verifier_api_key == "None":
                verifier_api_key = None
            result = compute_score_proof(
                proof_output=req.response,
                problem=req.question,
                reviewer=req.reviewer or "standard",
                reviews=req.reviews or 3,
                model_port=MODEL_PORT,
                model_name=verifier_model,
                api_key=verifier_api_key,
                marking=req.marking,  # 传递 marking 参数
            )
        else:
            if req.label is None:
                raise HTTPException(
                    status_code=400,
                    detail="`label` is required for standard scoring. Did you mean to set `is_proof` to true?",
                )
            # 直接同步调用，避免signal问题
            result = compute_score_p1(
                model_output=req.response, 
                label=req.label, 
                points=req.points, 
                question=req.question, 
                use_xverify=req.use_xverify,
                model_port=MODEL_PORT
            )
        return result
    except Exception as e:
        print(f"Error processing request: {e}")
        # 强制垃圾回收以清理可能的资源泄漏
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.on_event("startup")
async def startup_event():
    print("Reward model server started successfully!")
    # 设置信号处理器
    def signal_handler(signum, frame):
        print(f"Received signal {signum}, shutting down gracefully...")
        shutdown_event.set()
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

@app.on_event("shutdown")
async def shutdown_event_handler():
    print("Shutting down reward model server...")
    # 强制垃圾回收
    gc.collect()
    # 等待一小段时间确保资源释放
    await asyncio.sleep(1)

# 定期清理资源的后台任务
@app.middleware("http")
async def cleanup_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        # 每10个请求后强制垃圾回收
        if hasattr(request.app.state, 'request_count'):
            request.app.state.request_count += 1
        else:
            request.app.state.request_count = 1
        
        if request.app.state.request_count % 10 == 0:
            gc.collect()
            print(f"Cleaned up resources after {request.app.state.request_count} requests")
        
        return response
    except Exception as e:
        print(f"Error in middleware: {e}")
        # 强制垃圾回收
        gc.collect()
        raise

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Reward Model Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--timeout-keep-alive", type=int, default=30, help="Keep alive timeout")
    parser.add_argument("--log-level", default="info", help="Log level")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    print(f"Starting reward model server on {args.host}:{args.port}")
    uvicorn.run(
        "reward_model_server:app",
        host=args.host,
        port=args.port,
        timeout_keep_alive=args.timeout_keep_alive,
        log_level=args.log_level
    )
