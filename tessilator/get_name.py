def get_name_target(t_target):
    name_target = t_target.replace(" ", "_")
    name_spl = name_target.split("_")
    if name_spl[0] == 'Gaia':
        name_target = name_spl[-1]
    return name_target
