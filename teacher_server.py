import grpc
from concurrent import futures
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import pickle
from logits_service_pb2 import LogitsResponse
import logits_service_pb2_grpc
import base64
from typing import Dict, Any

MAX_MESSAGE_SIZE = 1000 * 1024 * 1024

class TeacherModelService:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        print(f"Teacher model loaded on {device}")

    def get_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get logits from teacher model"""
        with torch.no_grad():
            inputs = {
                'input_ids': input_ids.to(self.device),
                'attention_mask': attention_mask.to(self.device)
            }
            outputs = self.model(**inputs)
            print(outputs.logits.cpu())
            return outputs.logits.cpu()

    def serialize_tensor(self, tensor: torch.Tensor) -> str:
        tensor_bytes = pickle.dumps(tensor)
        return base64.b64encode(tensor_bytes).decode('utf-8')

    def deserialize_tensor(self, tensor_str: str) -> torch.Tensor:
        tensor_bytes = base64.b64decode(tensor_str.encode('utf-8'))
        return pickle.loads(tensor_bytes)

class LogitsServiceServicer:
    def __init__(self, teacher_service: TeacherModelService):
        self.teacher_service = teacher_service

    def GetLogits(self, request, context):
        try:
            input_ids = self.teacher_service.deserialize_tensor(request.input_ids)
            attention_mask = self.teacher_service.deserialize_tensor(request.attention_mask)
            logits = self.teacher_service.get_logits(input_ids, attention_mask)
            
            logits_str = self.teacher_service.serialize_tensor(logits)
            
            return LogitsResponse(
                logits=logits_str,
                success=True,
                error_message=""
            )
        except Exception as e:
            return LogitsResponse(
                logits="",
                success=False,
                error_message=str(e)
            )

def serve_teacher_model(model_name: str, port: int = 50051):
    teacher_service = TeacherModelService(model_name)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[('grpc.max_send_message_length', 512 * 1024 * 1024), ('grpc.max_receive_message_length', 512 * 1024 * 1024)])    
    servicer = LogitsServiceServicer(teacher_service)
    logits_service_pb2_grpc.add_LogitsServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    print(f"Teacher model server started on port {port}")
    server.wait_for_termination()

if __name__ == "__main__":
    serve_teacher_model("meta-llama/Llama-3.2-3B", port=50051)
