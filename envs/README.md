# Naming convention

Everything related to multiple agents for single environment step will be named with a singular word. Plural is only
used when referring to multiple environment steps. For example, reward for single step in both single and multi-agent
cases.

Annotated function types for jax transformed functions correspond to function types after the transformation.

Leading axis in an JAX array of single step is always the entity index.
Agent indices are always [0, num_agents-1]
Agent indices are prefix array of entity indices array.


