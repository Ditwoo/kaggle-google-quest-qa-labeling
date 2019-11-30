import argparse
import yaml
import torch
from src import registry

parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config', help='catalyst config file', required=True)
parser.add_argument('--state', dest='state', help='catalyst checkpoint file', required=True)
parser.add_argument('--out', dest='out', help='file to use for storing traced model', required=True)
parser.add_argument('--device', dest='device', help='device to use for tracing model', type=str, default='cpu')

config_file = 'configs/exp.yml'
model_file = 'validating/logs/checkpoints/last.pth'


def main():
    args = vars(parser.parse_args())
    config_file = args['config']
    model_file = args['state']
    out_file = args['out']
    device = torch.device(args['device'])

    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = registry.MODELS.get_from_params(**config['model_params'])
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    example_input = (
        torch.randint(high=1, size=(1, 25)).to(device),
        torch.randint(high=1, size=(1, 100)).to(device),
        torch.randint(high=1, size=(1, 50)).to(device),
        torch.randint(high=1, size=(1, 1)).to(device),
        torch.randint(high=1, size=(1, 1)).to(device),
    )
    trace = torch.jit.trace(model, example_input)
    torch.jit.save(trace, out_file)
    print(f'Traced model ({model_file}) to \'{out_file}\'', flush=True)


if __name__ == '__main__':
    main()
