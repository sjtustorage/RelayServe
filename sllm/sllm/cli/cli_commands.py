# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import click

from sllm.cli.cli_utils import (
    delete_model,
    deploy_model,
    show_status,
    start_server,
)


@click.group()
def cli():
    """Unified CLI for ServerlessLLM."""
    pass


@cli.command()
@click.option(
    "--model",
    help="Model name used as identifier (required if not specified in config file)",
)
@click.option(
    "--model-path",
    "model_path",
    default=None,
    help="Path or HuggingFace repo used by the backend to load weights (defaults to --model if not set)",
)
@click.option("--config", help="Path to configuration file")
@click.option("--backend", help="Backend framework (e.g., vllm, transformers)")
@click.option(
    "--num-gpus", type=int, help="Number of GPUs to use for the model"
)
@click.option("--target", type=int, help="Target number of requests per second")
@click.option(
    "--min-instances", type=int, help="Minimum number of model instances"
)
@click.option(
    "--max-instances", type=int, help="Maximum number of model instances"
)
@click.option(
    "--precision",
    help="Model precision for quantization (e.g., int8, fp4, nf4)",
)
@click.option(
    "--placement-node-id",
    "placement_node_id",
    default=None,
    help="Logical worker node id to pin both CPU and GPU instances to "
    "(sets backend_config.cpu_placement_node_id). Useful for multi-model "
    "benchmarks where replicas must be co-located on a specific node.",
)
@click.option(
    "--shared-cpu-model",
    "shared_cpu_model",
    default=None,
    help="Name of an already-deployed model whose CPU backend this model "
    "should reuse. The named model must be deployed first and must use "
    "the same model weights (same model_path). Saves one CPU backend "
    "instance per node in multi-replica scenarios.",
)
def deploy(
    model,
    model_path,
    config,
    backend,
    num_gpus,
    target,
    min_instances,
    max_instances,
    precision,
    placement_node_id,
    shared_cpu_model,
):
    """Deploy a model using a config file or model name.

    Either --model or a config file with a model specified is required.
    Command line options override values from the config file.
    """

    deploy_model(
        model=model,
        model_path=model_path,
        config=config,
        backend=backend,
        num_gpus=num_gpus,
        target=target,
        min_instances=min_instances,
        max_instances=max_instances,
        precision=precision,
        placement_node_id=placement_node_id,
        shared_cpu_model=shared_cpu_model,
    )


@cli.command()
@click.argument("models", nargs=-1)
def delete(models):
    """Delete deployed models."""
    delete_model(models)


@cli.command()
@click.option(
    "--host",
    default="0.0.0.0",
    type=str,
    help="Host IP to run the server on.",
)
@click.option(
    "--port", default=8343, type=int, help="Port to run the server on."
)
@click.option(
    "--hydraserve",
    is_flag=True,
    help="Enable HydraServe baseline: paired TP/PP prewarm, solver-free cold load, load_async=off in GPU engine.",
)
@click.option(
    "--serverlessllm",
    is_flag=True,
    help="Enable serverlessllm paper baseline: one prewarmed empty GPU (lazy_load+load_async), cold: full lazy_init then inference, then prewarm again.",
)
@click.option(
    "--vllm",
    is_flag=True,
    help="Enable plain vLLM mode: weights loaded directly from disk (load_format=auto), no shared-memory KV transfer.",
)
@click.option(
    "--static",
    "static",
    is_flag=True,
    help="Ablation: skip solver; evaluate both static-tokenwise and static-layerwise candidates and pick the one with lower estimated TTFT.",
)
@click.option(
    "--original-amx",
    "original_amx",
    is_flag=True,
    help="Start CPU backend with VLLM_CPU_USE_AMX_GEMM_BF16=0 and use original-AMX perf profile for solver.",
)
def start(host, port, hydraserve, serverlessllm, vllm, static, original_amx):
    """Start the head node of the SLLM cluster."""
    start_server(
        host=host,
        port=port,
        hydraserve=hydraserve,
        serverlessllm=serverlessllm,
        vllm=vllm,
        static=static,
        original_amx=original_amx,
    )


@cli.command()
def status():
    """Show all deployed models."""
    show_status()


if __name__ == "__main__":
    cli()
