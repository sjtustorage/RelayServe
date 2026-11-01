"""Global-WORLD rank endpoint manifest for the resident HTTP controller."""
from pathlib import Path


def parse_cluster_config(config):
    if not isinstance(config, dict) or config.keys() != {'key_file', 'ranks'}:
        raise ValueError('Cluster config requires key_file and ranks')
    if not isinstance(config['key_file'], str) or not config['key_file']:
        raise ValueError('Cluster key_file must be a nonempty path')
    ranks = config['ranks']
    if not isinstance(ranks, list) or not ranks:
        raise ValueError('Cluster ranks must be a nonempty global-rank ordered list')
    endpoints, nodes = [], []
    for rank in ranks:
        if (not isinstance(rank, dict) or rank.keys() != {'host', 'port', 'node'}
                or not isinstance(rank['host'], str) or not rank['host']
                or type(rank['port']) is not int or not 1 <= rank['port'] <= 65535
                or not isinstance(rank['node'], str) or not rank['node']):
            raise ValueError('Each rank requires host, port and physical node')
        endpoints.append((rank['host'], rank['port']))
        nodes.append(rank['node'])
    if len(set(endpoints)) != len(endpoints):
        raise ValueError('Cluster rank endpoints must be unique')
    key = Path(config['key_file']).read_text().strip()
    if not key:
        raise ValueError('Cluster control key must be nonempty')
    return endpoints, nodes, key
