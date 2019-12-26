def get_parameter(parameters: dict,
                  key: str,
                  mandatory: bool = True,
                  delete: bool = False,
                  default_value=None):
    """Gets a single parameter from a parameters dictionary

    Args:
        parameters (dict): parameters dictionary
        key (str): parameter key to be retrieved
        mandatory (bool): raise an error if parameter is not found
        delete (bool): delete value from parameters dictionary after get
        default_value: default value, if parameter is not found (and not mandatory)

    """
    key_split = key.split('.', 1)
    if len(key_split) == 2:
        sub_parameters = get_parameter(parameters=parameters,
                                       key=key_split[0],
                                       mandatory=mandatory,
                                       delete=False)

        if sub_parameters is not None:
            return get_parameter(parameters=sub_parameters,
                                 key=key_split[1],
                                 mandatory=mandatory,
                                 delete=delete,
                                 default_value=default_value)

    if key in parameters:
        value = parameters.get(key)
        if delete:
            del parameters[key]
        return value
    else:
        if mandatory:
            raise RuntimeError('parameter \'{}\' is not in parameters list'.format(key))
        else:
            return default_value


def extract_parameter(parameters: dict,
                      key: str,
                      mandatory: bool = True,
                      default_value=None):
    """Extracts a particular parameter from a parameters dictionary
       deleting the parameter from the dictionary

    Args:
        parameters (dict): parameter dictionary
        key (str): parameter key to be extracted
        mandatory (bool): raise an error if parameter is not found
        default_value:

    """
    return get_parameter(parameters=parameters,
                         key=key,
                         mandatory=mandatory,
                         delete=True,
                         default_value=default_value)


def get_parameters(parameters: dict,
                   key_list: list,
                   delete_parameter: bool = False) -> dict:
    """Get the list of parameter values by its keys

    Args:
        parameters: parameters dictionary
        key_list: list of parameters to be retrieved
        delete_parameter: delete parameter from parameters dictionary after retrieval

    """
    parameter_values = {}
    for key in key_list:
        if key not in parameters:
            parameter_values[key] = None
            continue

        parameter_values[key] = parameters.get(key)
        if delete_parameter:
            del parameters[key]

    return parameter_values


def extract_parameters(parameters: dict, key_list: list) -> dict:
    """Extracts a list of parameter values from a parameters dictionary
       deleting from the original parameter dictionary

    Args:
        parameters (dict): parameter dictionary
        parameter_list (list): parameter key list to be extracted

    """
    return get_parameters(parameters=parameters,
                          key_list=key_list,
                          delete_parameter=True)
