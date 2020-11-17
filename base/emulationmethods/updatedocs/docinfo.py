[emulationfitinfo]
""" 
This [emulationfitinfo] automatically filled by docinfo.py when running updatedocs.py
The purpose of automatic documentation is to ease the burden for new methods.
##############################################################################
################################### fit ######################################
### The purpose of this is to take information and plug all of our fit
### information into fitinfo, which is a python dictionary. 
##############################################################################
##############################################################################
"""


[emulationfitdocstring]
    r"""
    Fits a emulation model.
    This [emulationfitdocstring] automatically filled by docinfo.py when running updatedocs.py
    
    Parameters
    ----------
    fitinfo : dict
        An arbitary dictionary where you should place all of your fitting information once complete.
        This dictionary is pass by reference, so there is no reason to return anything. Keep
        only stuff that will be used by predict below.  Note that if you want to leverage speedy,
        updates fitinfo will contain the previous fitinfo!  So use that information to accelerate 
        anything you wany, keeping in mind that the user might have changed theta, f, and x.
    x : array of objects
        An matrix (vector) of inputs. Each row should correspond to a row in f.
    theta :  array of objects
        An matrix (vector) of parameters. Each row should correspond to a row in f.
    f : array of float
        An matrix (vector) of responses.  Each row in f should correspond to a row in x. Each 
        column should correspond to a row in theta.
    args : dict
        A dictionary containing options passed to you.
    """


[emulationpredictinfo]
"""
This [emulationpredictinfo] automatically filled by docinfo.py when running updatedocs.py
##############################################################################
################################### predict ##################################
### The purpose of this is to take an emulator emu alongside fitinfo, and 
### predict at x and theta. You shove all your information into the dictionary predinfo.
##############################################################################
##############################################################################
"""

[emulationpredictdocstring]
    r"""
    Finds prediction at theta and x given the dictionary fitinfo.
    This [emulationpredictdocstring] automatically filled by docinfo.py when running updatedocs.py

    Parameters
    ----------
    predinfo : dict
        An arbitary dictionary where you should place all of your prediction information once complete. 
        This dictionary is pass by reference, so there is no reason to return anything. Keep
        only stuff that will be used by predict.  Key elements
        predinfo['mean'] : predinfo['mean'][k] is mean of the prediction at all x at theta[k].
        predinfo['var'] : predinfo['var'][k] is variance of the prediction at all x at theta[k].
        predinfo['cov'] : predinfo['cov'][k] is mean of the prediction at all x at theta[k].
        predinfo['covhalf'] : if A = predinfo['covhalf'][k] then A.T @ A = predinfo['cov'][k]
        predinfo['rand'] : predinfo['rand'][l][k] lth draw of of x at theta[k].
    fitinfo : dict
        An arbitary dictionary where you placed all your important fitting information from the 
        fit function above.
    x : array of objects
        An matrix (vector) of inputs for prediction.
    theta :  array of objects
        An matrix (vector) of parameters to prediction.
    args : dict
        A dictionary containing options passed to you.
    """

[emulationadditionalfuncsinfo]
"""
This [emulationadditionalfuncsinfo] automatically filled by docinfo.py when running updatedocs.py
##############################################################################
##############################################################################
####################### THIS ENDS THE REQUIRED PORTION #######################
###### THE NEXT FUNCTIONS ARE OPTIONAL TO BE CALLED BY CALIBRATION ###########
## If this project works, there will be a list of useful calibration functions
## to provide as you want.
##############################################################################
##############################################################################
"""

[emulationsupplementdocstring]
    r"""
    Finds supplement theta and x given the dictionary fitinfo.
    This [emulationsupplementdocstring] is automatically filled by docinfo.py when running updatedocs.py

    Parameters
    ----------
    fitinfo : dict
        An arbitary dictionary where you placed all your important fitting information from the 
        fit function above.
    x : array
        An array of x values where you want to predict.
    theta : array
        An array of theta values where you want to predict.
    cal : instance of emulator class
        An emulator class instance as defined in calibration.  This will not always be provided.
    args : dict
        A dictionary containing options passed to you.
        
    Returns
    ----------
    Note that we should have theta.shape[0] * x.shape[0] < size
    theta : array
        An array of theta values where you should sample.
    x : array
        An array of x values where you should sample.
    info : array
        An an optional info dictionary you can pass back to the user.
    """

[endfunctionsflag]
"""
This [endfunctionsflag] is automatically filled by docinfo.py when running updatedocs.py
##############################################################################
##############################################################################
####################### THIS ENDS THE OPTIONAL PORTION #######################
######### USE SPACE BELOW FOR ANY SUPPORTING FUNCTIONS YOU DESIRE ############
##############################################################################
##############################################################################
"""
