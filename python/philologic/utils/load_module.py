#!/var/lib/philologic5/philologic_env/bin/python3
"""Load Python source file"""

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader


def load_module(module_name, path):
    """Load arbitrary Python source file"""
    loader = SourceFileLoader(module_name, path)
    spec = spec_from_loader(loader.name, loader)
    module = module_from_spec(spec)
    loader.exec_module(module)
    return module
