class BiseModuleContainer:
    """We need this class to ensure that in torch.nn.Module, the bise_module does not become one of its child.
    """
    def __init__(self, bise_module, *args, **kwargs) -> None:
        self.bise_module = bise_module
