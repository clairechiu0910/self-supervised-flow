import json
from train import train, set_seed
from dataset.dataset import prepare_dataloaders
from utils import write_verbose
from config import get_args, get_init_parameters

c = get_args()
get_init_parameters(c)

set_seed(c.seed)

if c.verbose:
    write_verbose(c.model_name, json.dumps(c.__dict__))
    write_verbose(c.model_name, c.murmur)

train_loader, test_loader = prepare_dataloaders(c)
train(c, train_loader, test_loader)
