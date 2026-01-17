import os
import sys
import random
import errno
import time
import torch
import numpy as np
from datetime import timedelta
import signal

class AverageMeter:
    """
    Computes and stores the average and current value.
    Used to track metrics like loss, accuracy, etc.
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset all statistics to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        """
        Update statistics with a new value.
        
        Args:
            val (float): New value to be added to the statistics.
        """
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

def setup_system(seed, cudnn_benchmark=True, cudnn_deterministic=True) -> None:
    """
    Set seeds for reproducible training across different components.
    
    Args:
        seed (int): Random seed to be used.
        cudnn_benchmark (bool): Whether to enable cuDNN benchmarking.
        cudnn_deterministic (bool): Whether to enable cuDNN deterministic mode.
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn_benchmark_enabled = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
      
def mkdir_if_missing(dir_path):
    """
    Create directory if it doesn't exist.
    
    Args:
        dir_path (str): Directory path to be created.
    """
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def handle_signals():
    """
    Set up signal handlers for graceful program termination.
    Handles SIGINT (Ctrl+C) and SIGTERM signals.
    """
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

class Logger(object):
    """
    Logger class that can write to both console and file simultaneously.
    
    Args:
        fpath (str, optional): Path to the log file. If None, only console output is used.
        auto_flush (bool): Whether to automatically flush after each write operation.
    """
    def __init__(self, fpath=None, auto_flush=False):
        self.console = sys.stdout
        self.file = None
        self.auto_flush = auto_flush
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w', buffering=1)
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.close()
    
    def close(self):
        """Close all open file handles."""
        try:
            if self.file is not None:
                self.flush()
                self.file.close()
                self.file = None
            if hasattr(self.console, 'close'):
                self.console.close()
        except Exception as e:
            print(f"Error closing logger: {e}")

    def write(self, msg):
        """
        Write message to both console and file.
        
        Args:
            msg (str): Message to be written.
        """
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)
            if self.auto_flush:
                self.flush()

    def flush(self):
        """Flush both console and file buffers."""
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

def sec_to_min(seconds):
    """
    Convert seconds to minutes:seconds format.
    
    Args:
        seconds (int): Number of seconds to convert.
        
    Returns:
        str: Time in MM:SS format.
    """
    seconds = int(seconds)
    minutes = seconds // 60
    seconds_remaining = seconds % 60
    
    if seconds_remaining < 10:
        seconds_remaining = '0{}'.format(seconds_remaining)
    
    return '{}:{}'.format(minutes, seconds_remaining)

def sec_to_time(seconds):
    """
    Convert seconds to HH:MM:SS format.
    
    Args:
        seconds (int): Number of seconds to convert.
        
    Returns:
        str: Time in HH:MM:SS format.
    """
    return "{:0>8}".format(str(timedelta(seconds=int(seconds))))

def print_time_stats(t_train_start, t_epoch_start, epochs_remaining, steps_per_epoch):
    """
    Print training time statistics including elapsed time, epoch speed, and ETA.
    
    Args:
        t_train_start (float): Training start timestamp.
        t_epoch_start (float): Current epoch start timestamp.
        epochs_remaining (int): Number of epochs remaining.
        steps_per_epoch (int): Number of steps per epoch.
    """
    elapsed_time = time.time() - t_train_start
    speed_epoch = time.time() - t_epoch_start 
    speed_batch = speed_epoch / steps_per_epoch
    eta = speed_epoch * epochs_remaining
        
    print("Elapsed {}, {} time/epoch, {:.2f} s/batch, remaining {}".format(
                sec_to_time(elapsed_time), sec_to_time(speed_epoch), speed_batch, sec_to_time(eta)))
    
