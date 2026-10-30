"""Set UCX's error handling mode for every Python process that imports it.

NIXL defaults to "peer", which needs tcp or RDMA. UCX finds neither on Modal, so
stock vLLM cannot start its NIXL backend. vLLM has no setting for this yet, and
PYTHONPATH reaches its spawned workers too. With "none", a crashed peer may hang
a transfer instead of failing it, so keep this to experiments.
"""

from nixl._api import nixl_agent

create_backend = nixl_agent.create_backend


def with_error_mode(self, backend, initParams={}):
    if backend == "UCX":
        initParams = {**initParams, "ucx_error_handling_mode": "none"}
    return create_backend(self, backend, initParams)


nixl_agent.create_backend = with_error_mode
