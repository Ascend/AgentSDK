"""Read optional runtime settings; old YAMLs keep their existing defaults."""
import sys
import yaml


def runtime_options(config, default_cards):
    cards = config.get('npu_num_per_node', default_cards)
    if type(cards) is not int or cards <= 0:
        raise ValueError('npu_num_per_node must be a positive integer')
    eager = config.get('enforce_eager', False)
    if type(eager) is not bool:
        raise ValueError('enforce_eager must be a YAML boolean')
    if eager and config.get('use_vllm_opt', False):
        raise ValueError('enforce_eager requires use_vllm_opt: false')
    return cards, str(eager).lower()


if __name__ == '__main__':
    with open(sys.argv[1], encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    print(*runtime_options(config, int(sys.argv[2])))
