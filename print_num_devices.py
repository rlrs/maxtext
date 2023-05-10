import jax
import socket
from jax.experimental import multihost_utils
import portpicker
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--process_count', type=int)
parser.add_argument('--process_index', type=int)
args = parser.parse_args()

def gen_local_ip():
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

def gen_local_ip_nums():
    return [int(num) for num in gen_local_ip().split(':')[-1].split('.')]

def get_coordinator_ip():
    local_ip_nums = jax.numpy.array(gen_local_ip_nums())
    coordinator_ip_nums = multihost_utils.broadcast_one_to_all(local_ip_nums)
    coordinator_ip_strings = [str(num) for num in list(coordinator_ip_nums)]
    return '.'.join(coordinator_ip_strings)

#port = multihost_utils.broadcast_one_to_all(jax.numpy.array(portpicker.pick_unused_port()))
#coordinator_address = get_coordinator_ip() + ':' + str(port)
coordinator_address = "10.128.0.15:23583"
jax.distributed.initialize(coordinator_address=coordinator_address,
                            num_processes=args.process_count,
                            process_id=args.process_index)

print("jax.process_index():", jax.process_index())
print("jax.process_count():", jax.process_count())
print("jax.host_id():", jax.host_id())
print("jax.host_count():", jax.host_count())
print("jax.local_device_count():", jax.local_device_count())
print("jax.device_count():", jax.device_count())
