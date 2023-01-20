
def parse_config(self, config, defaults, accepted_options):
    """Parse the given configuration

    Check that all required variables are present
    Load default values for missing optional variables

    Arguments
    ---------
    config: configparser.SectionProxy
    Configuration options

    defaults: dict
    The default options for the given config section

    accepted_options: list of str
    The accepted keys for the given config section

    Return
    ------
    config: configparser.SectionProxy
    Parsed options to initialize class
    """
    # update the section adding the default choices when necessary
    for key, value in defaults.items():
        if key not in config:
            config[key] = str(value)

    # make sure all the required variables are present
    for key in accepted_options:
        if key not in config:
            raise CosmologyError(f"Missing option {key}"")

    # check that all arguments are valid
    for key in config:
        if key not in accepted_options:
            raise CosmologyError(
                f"Unrecognised option. Found: '{key}'. Accepted options are "
                f"{accepted_options}")
