#!/bin/bash
# Launch script for AlphaZero Kalah training on 4x A100 GPUs

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create necessary directories
mkdir -p kalah_data
mkdir -p kalah_checkpoints
mkdir -p kalah_logs
mkdir -p kalah_tensorboard

# Function to check GPU availability
check_gpus() {
    echo "Checking GPU availability..."
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ $gpu_count -lt 4 ]; then
        echo "WARNING: Less than 4 GPUs detected. Found $gpu_count GPUs."
        echo "Do you want to continue? (y/n)"
        read -r response
        if [ "$response" != "y" ]; then
            exit 1
        fi
    fi
}

# Function to launch training
launch_training() {
    echo "Starting AlphaZero Kalah training..."
    echo "Configuration:"
    echo "  - GPUs: $CUDA_VISIBLE_DEVICES"
    echo "  - Log directory: ./kalah_logs"
    echo "  - Checkpoint directory: ./kalah_checkpoints"
    echo ""
    
    # Launch training with proper error handling
    # python main.py "$@" 2>&1 | tee kalah_logs/training_$(date +%Y%m%d_%H%M%S).log
    python -m cProfile -s cumulative main.py > main_0.txt
}

# Function to launch monitoring
launch_monitoring() {
    echo "Starting monitoring server..."
    python monitor.py --web --port 8000 &
    MONITOR_PID=$!
    echo "Monitor PID: $MONITOR_PID"
    echo "Web interface available at http://localhost:8000"
}

# Parse command line arguments
RESUME=""
CONFIG=""
MONITOR=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME="--resume $2"
            shift 2
            ;;
        --config)
            CONFIG="--config $2"
            shift 2
            ;;
        --monitor)
            MONITOR=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--resume checkpoint_path] [--config config_file] [--monitor]"
            exit 1
            ;;
    esac
done

# Main execution
echo "="*60
echo "AlphaZero Kalah Training Launcher"
echo "="*60

# Check GPUs
check_gpus

# Launch monitoring if requested
if [ "$MONITOR" = true ]; then
    launch_monitoring
    sleep 2
fi

# Launch training
launch_training $RESUME $CONFIG

# Cleanup
if [ "$MONITOR" = true ] && [ -n "$MONITOR_PID" ]; then
    echo "Stopping monitor..."
    kill $MONITOR_PID
fi

echo "Training completed!"