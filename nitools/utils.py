def extract_kwargs_from_ctx(ctx):
    """ Extracts kwargs from Click context manager. """
    args = []
    i = 0
    for arg in ctx.args:
        if arg[:2] == '--':
            args.append([arg[2:]])
            i += 1
        else:
            args[(i-1)].append(arg)

    for i, arg in enumerate(args):
        if len(arg) == 1:
            args[i] = [arg[0], True]
        elif len(args) > 2:
            args[i] = [arg[0], ' '.join(arg[1:])]

    keys = [arg[0] for arg in args]
    if len(keys) != len(set(keys)):
        msg = "Your cmd arguments contain a duplicate!"
        raise ValueError(msg)

    kwargs = {arg[0]: arg[1] for arg in args}
    return kwargs

