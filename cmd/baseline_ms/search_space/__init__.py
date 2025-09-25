def init_search_space(space_name: str, args):
    if space_name == "mlp":
        from .mlp import MlpSpace, MlpMacroCfg, DEFAULT_LAYER_CHOICES_20
        model_cfg = MlpMacroCfg(
            args.nfield,
            args.nfeat,
            args.nemb,
            args.num_layers,
            args.num_labels,
            DEFAULT_LAYER_CHOICES_20)
        return MlpSpace(model_cfg)
    else:
        raise Exception
