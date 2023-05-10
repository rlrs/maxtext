import jax

jax.distributed.initialize("34.69.59.135:1234", num_processes=2, process_id=jax.process_index())

print(jax.device_count())
