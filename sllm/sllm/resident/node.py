"""One persistent GPU pool per node, controlled by an exclusive HTTP lease."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

from sllm.resident.client import ResidentPoolClient
from sllm.resident.remote import NodeAPI
from sllm.resident.server import make_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--torchrun', required=True)
    parser.add_argument('--gpu-count', required=True, type=int)
    parser.add_argument('--key-file', required=True, type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=8001, type=int)
    parser.add_argument('--timeout', default=120, type=float)
    parser.add_argument('--copy-throttle', action=argparse.BooleanOptionalAction,
                        default=True, help='Throttle weight DMA submissions during GPU initialization')
    parser.add_argument('--copy-throttle-tasks', type=int, default=32)
    parser.add_argument('--copy-throttle-first-only', action='store_true')
    args = parser.parse_args()
    key = args.key_file.read_text().strip()
    if not key:
        parser.error('key-file must contain a nonempty node control key')
    with ExitStack() as owners:
        pool = owners.enter_context(ResidentPoolClient(args.torchrun, args.gpu_count, timeout=args.timeout,
                            copy_throttle=args.copy_throttle,
                            copy_throttle_tasks=args.copy_throttle_tasks,
                            copy_throttle_first_only=args.copy_throttle_first_only))
        pool.command('status')  # Complete process/CUDA startup before listening.
        api_holder = []
        def rebuild_pool():
            return owners.enter_context(api_holder[0].pool.replacement())
        api = NodeAPI(pool, key, pool_factory=rebuild_pool)
        api_holder.append(api)
        server = make_server((args.host, args.port), api)
        print(json.dumps(dict(listening=server.server_address, gpu_count=args.gpu_count,
                              copy_throttle=args.copy_throttle)), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == '__main__':
    main()
