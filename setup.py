import subprocess
import os

def setup_grpc():
    """Generate gRPC files from proto definition"""
    
    # Create proto file
    proto_content = '''
syntax = "proto3";

package logits;

service LogitsService {
    rpc GetLogits(LogitsRequest) returns (LogitsResponse);
}

message LogitsRequest {
    string input_ids = 1;
    string attention_mask = 2;
    string model_config = 3;
}

message LogitsResponse {
    string logits = 1;
    bool success = 2;
    string error_message = 3;
}
'''
    
    with open('logits_service.proto', 'w') as f:
        f.write(proto_content)
    
    # Generate Python files
    subprocess.run([
        'python', '-m', 'grpc_tools.protoc',
        '--proto_path=.',
        '--python_out=.',
        '--grpc_python_out=.',
        'logits_service.proto'
    ])
    
    print("gRPC files generated successfully!")

if __name__ == "__main__":
    setup_grpc()
