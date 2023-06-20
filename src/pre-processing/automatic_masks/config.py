INCLUDE_SURFACE_WATER = True
INCLUDE_GLACIER = True
INCLUDE_EXO_LABS = True
PROCESS_S2_CLOUDLESS = True


# ====================================
# ====================================
# Some helper functions
# ====================================
# ====================================

def report_config():
    """Report the configuration of the model."""

    print(f"\nConfiguration:")
    for var in globals():

        # check if it is a variable and not a function
        if not var.startswith('__') and not callable(var) and var == var.upper():
            print(f" - {var} = {eval(var)}")

    print(f"\n\n")

