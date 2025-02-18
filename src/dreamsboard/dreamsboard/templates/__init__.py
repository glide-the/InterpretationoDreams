import os

from dreamsboard.common.registry import registry

root_dir = os.path.dirname(os.path.abspath(__file__))
registry.register_path("templates_library_root", root_dir)


def get_template_path(name):
    return os.path.join(registry.get_path("templates_library_root"), name)
