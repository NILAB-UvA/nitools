def extract_kwargs_from_ctx(ctx):
    """ Extracts kwargs from Click context manager. """
    kwargs = {}
    i = 0
    while True:
        if ctx.args[i][:2] == '--':
            key = ctx.args[i][2:]
            if (i+1) == len(ctx.args):
                kwargs[key] = True
                break
            if ctx.args[i+1][:2] == '--':
                value = True
            else:
                value = ctx.args[i+1]
            kwargs[key] = value
        i += 2
        if i >= len(ctx.args):
            break
    return kwargs

