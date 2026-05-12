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
import logging
from concurrent.futures import ThreadPoolExecutor
from p1 import compute_score_p1, Model_args
from proof_verifier import compute_score_proof
import proof_verifier

app = FastAPI()
# Reuse uvicorn's configured logger so INFO logs are emitted reliably.
logger = logging.getLogger("uvicorn.error")

# Global variable for graceful shutdown
shutdown_event = asyncio.Event()

# Global variable for storing model port
MODEL_PORT = int(os.environ.get('MODEL_PORT', 34882))
MODEL_NAME = os.environ.get("MODEL_NAME")
PROOF_VERIFIER_MODEL_NAME = os.environ.get("PROOF_VERIFIER_MODEL_NAME")
PROOF_MAX_INFLIGHT = max(1, int(os.environ.get("PROOF_MAX_INFLIGHT", "32")))
PROOF_THREAD_WORKERS = max(1, int(os.environ.get("PROOF_THREAD_WORKERS", str(PROOF_MAX_INFLIGHT))))
REQUEST_LOG_EVERY = max(1, int(os.environ.get("REQUEST_LOG_EVERY", "20")))

class RewardRequest(BaseModel):
    response: str
    label: Optional[Union[str, List[str]]] = None  # Can be None
    points: Optional[List[float]] = None
    question: Optional[str] = None
    use_xverify: bool = False
    is_proof: bool = False
    reviewer: Optional[str] = None
    reviews: Optional[int] = None


async def _run_proof_task(func, *args, **kwargs):
    """Run blocking proof verification in thread pool with inflight control."""
    async with app.state.proof_semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            app.state.proof_executor,
            lambda: func(*args, **kwargs),
        )


@app.post("/")
async def evaluate_reward(req: RewardRequest):
    # print(f"Received request payload: {req}")
    try:
        started_at = time.time()
        is_proof_request = req.is_proof or (req.label is None and req.points is None)
        if not is_proof_request:
            logger.warning(
                "This server is intended only for proof requests; request may be misrouted"
            )
        if not req.question:
            raise HTTPException(status_code=400, detail="`question` is required for proof verification")
        selected_reviewer = req.reviewer or "ds_proof"
        verifier_model = (
            PROOF_VERIFIER_MODEL_NAME
            or MODEL_NAME
            or (Model_args.model_name if hasattr(Model_args, "model_name") else None)
        )
        verifier_api_key = getattr(Model_args, "api_key", None)
        if verifier_api_key == "None":
            verifier_api_key = None
        result = await _run_proof_task(
            compute_score_proof,
            proof_output=req.response,
            problem=req.question,
            reviewer=selected_reviewer,
            reviews=req.reviews or 1,
            model_port=MODEL_PORT,
            model_name=verifier_model,
            api_key=verifier_api_key,
        )
        app.state.proof_request_seq += 1
        request_seq = app.state.proof_request_seq
        model_stats = result.get("model_stats", {}) if isinstance(result, dict) else {}
        if selected_reviewer == "ds_proof" and model_stats.get("finish_reason") == "error":
            logger.warning(
                "proof_request error seq=%d reviewer=%s elapsed_ms=%.1f question_chars=%d response_chars=%d "
                "model_output_chars=%s error=%s",
                request_seq,
                selected_reviewer,
                (time.time() - started_at) * 1000,
                len(req.question or ""),
                len(req.response or ""),
                model_stats.get("output_char_len"),
                model_stats.get("error"),
            )
        if selected_reviewer == "ds_proof" and request_seq % REQUEST_LOG_EVERY == 0:
            logger.info(
                "proof_request seq=%d reviewer=%s score=%s elapsed_ms=%.1f question_chars=%d response_chars=%d "
                "model_output_chars=%s truncated=%s finish_reason=%s completion_tokens=%s error=%s",
                request_seq,
                selected_reviewer,
                result.get("score") if isinstance(result, dict) else "n/a",
                (time.time() - started_at) * 1000,
                len(req.question or ""),
                len(req.response or ""),
                model_stats.get("output_char_len"),
                model_stats.get("truncated"),
                model_stats.get("finish_reason"),
                model_stats.get("completion_tokens"),
                model_stats.get("error"),
            )
        #     if req.label is None:
        #         raise HTTPException(
        #             status_code=400,
        #             detail="`label` is required for standard scoring. Did you mean to set `is_proof` to true?",
        #         )
        #     # Direct synchronous call to avoid signal issues
        #     result = compute_score_p1(
        #         model_output=req.response, 
        #         label=req.label, 
        #         points=req.points, 
        #         question=req.question, 
        #         use_xverify=req.use_xverify,
        #         model_port=MODEL_PORT
        #     )
        return result
    except Exception as e:
        logger.exception("Error processing request: %s", e)
        # Force garbage collection to clean up possible resource leaks
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.on_event("startup")
async def startup_event():
    logger.info("Reward model server started successfully!")
    logger.info("Using proof_verifier module: %s", getattr(proof_verifier, "__file__", "unknown"))
    app.state.proof_semaphore = asyncio.Semaphore(PROOF_MAX_INFLIGHT)
    app.state.proof_executor = ThreadPoolExecutor(
        max_workers=PROOF_THREAD_WORKERS,
        thread_name_prefix="proof-rm",
    )
    app.state.proof_request_seq = 0
    logger.info(
        "Proof verifier config: max_inflight=%d thread_workers=%d request_log_every=%d",
        PROOF_MAX_INFLIGHT,
        PROOF_THREAD_WORKERS,
        REQUEST_LOG_EVERY,
    )
    # Set signal handler
    def signal_handler(signum, frame):
        logger.info("Received signal %s, shutting down gracefully...", signum)
        shutdown_event.set()
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

@app.on_event("shutdown")
async def shutdown_event_handler():
    logger.info("Shutting down reward model server...")
    if hasattr(app.state, "proof_executor"):
        app.state.proof_executor.shutdown(wait=False, cancel_futures=True)
    # Force garbage collection
    gc.collect()
    # Wait for a short time to ensure resources are released
    await asyncio.sleep(1)

# Background task to periodically clean up resources
@app.middleware("http")
async def cleanup_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        # Force garbage collection after every 10 requests
        if hasattr(request.app.state, 'request_count'):
            request.app.state.request_count += 1
        else:
            request.app.state.request_count = 1
        
        if request.app.state.request_count % 10 == 0:
            gc.collect()
            logger.info("Cleaned up resources after %d requests", request.app.state.request_count)
        
        return response
    except Exception as e:
        logger.exception("Error in middleware: %s", e)
        # Force garbage collection
        gc.collect()
        raise

def parse_arguments():
    """Parse command line arguments"""
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
