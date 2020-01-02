import argparse
import yaml
import torch
from . import registry

parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config', help='catalyst config file', required=True)
parser.add_argument('--state', dest='state', help='catalyst checkpoint file', required=True)
parser.add_argument('--out', dest='out', help='file to use for storing traced model', required=True)
parser.add_argument('--device', dest='device', help='device to use for tracing model', type=str, default='cuda:0')
parser.add_argument('--input-type', dest='inp_type', help='model input type', type=str, default='transformers')


def main():
    args = vars(parser.parse_args())
    config_file = args['config']
    model_file = args['state']
    out_file = args['out']
    device = torch.device(args['device'])
    input_type = args['inp_type']

    if input_type not in {'transformers', 'fields', 'transformers-with-categories'}:
        raise ValueError("'--input-type' should be one of "
                         "('transformers', 'fields', 'transformers-with-categories')!")

    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = registry.MODELS.get_from_params(**config['model_params'])
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    if input_type == 'transformers':
        example_input = torch.randint(low=1, high=10, size=(1, 512)).to(device)
        example_input[0, 0] = 101
        example_seg = torch.randint(low=0, high=2, size=(1, 512)).to(device)
        model_input = (example_input, example_seg)
    elif input_type == 'fields':
        example_question_title = torch.randint(low=0, high=10, size=(1, 100)).to(device)
        example_question_body = torch.randint(low=0, high=10, size=(1, 200)).to(device)
        example_answer = torch.randint(low=0, high=10, size=(1, 250)).to(device)
        example_category = torch.randint(high=5, size=(1, 1)).to(device)
        example_host = torch.randint(high=5, size=(1, 1)).to(device)
        model_input = (example_question_title, example_question_body, example_answer, 
                       example_category, example_host)
    else:
        # transformers with categories
        example_input = torch.randint(low=1, high=10, size=(1, 512)).to(device)
        example_input[0, 0] = 101
        example_seg = torch.randint(low=0, high=2, size=(1, 512)).to(device)
        example_category = torch.randint(high=5, size=(1, 1)).to(device)
        example_host = torch.randint(high=5, size=(1, 1)).to(device)
        model_input = (example_input, example_category, example_host, example_seg)
        
    trace = torch.jit.trace(model, model_input)
    torch.jit.save(trace, out_file)
    print(f"Traced model (checkpoint - '{model_file}') to '{out_file}'", flush=True)


if __name__ == '__main__':
    main()
