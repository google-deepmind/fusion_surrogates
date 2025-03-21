import jax
import jax.numpy as jnp


def normalize(data: jax.Array, *, mean: jax.Array, stddev: jax.Array) -> jax.Array:
    """Normalizes data to have mean 0 and stddev 1."""
    return (data - mean) / jnp.where(stddev == 0, 1, stddev)


def unnormalize(data: jax.Array, *, mean: jax.Array, stddev: jax.Array) -> jax.Array:
    """Unnormalizes data to the orginal distribution."""
    return data * jnp.where(stddev == 0, 1, stddev) + mean
