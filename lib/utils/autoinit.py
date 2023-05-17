import inspect
from pkgutil import iter_modules
import os
import sys
import pickle
import glob

def get_modules():
    """
    returns the names of all modules in this project by getting a list of .py
    files from the base module from __loader__.path and __loader__.name
    """
    name = __loader__.name
    path = __loader__.path
    numdots = len(name.split("."))-1
    basemodule = path.split("/")[-numdots-1]
    basepath   = "/".join(path.split("/")[:-numdots])
    modules = [basemodule+'.'+f[len(basepath)+1:-3].replace("/", ".") for f in glob.glob(f'{basepath}/**/*.py')]
    return modules

def find_class(class_name):
    """
    tries to find the class_name using get_modules
    class_name: str
    returns: a class object
             or raises an exception if not found
    """
    for mod_name in get_modules():
        try:
            exec(f'from {mod_name} import {class_name}')
            return eval(f'{class_name}')
        except:
            pass

    raise ValueError(f"class '{class_name}' not found")

def autoinit(obj):
    """
    retrieves the args with which the calling function was invoked,
    adds them as variables into obj, and also adds dict 'init_args' gathering its values 
    """
    print ("autoinit on", obj.__class__.__name__)
    stacks = inspect.stack()
    cframe = stacks[1].frame
    args_info = inspect.getargvalues(cframe)
    for k in ['self', '__class__']:
        if k in args_info.locals.keys():
            args_info.locals.pop(k)
        
    args = args_info.locals
    obj.init_args = args

    for k,v in args.items():
        setattr(obj, k, v)

def get_autoinit_wandb(obj):
    """
    gets autoinit args of an object, replacing classes, methods and functions by their name
    """
    if not 'init_args' in dir(obj):
        raise ValueError(f"object {obj} was not initialized with @autoinit")

    wandb_conf = {}
    for k,v in obj.init_args.items():
        if inspect.ismethod(v):
            wandb_conf[k] = v.__qualname__        
        elif inspect.isclass(v) or inspect.isfunction(v):
            wandb_conf[k] = v.__qualname__
        else:
            wandb_conf[k] = v

    # add object specific wandb stuff
    if 'get_additional_wandb_config' in dir(obj):
        wandb_conf.update(obj.get_additional_wandb_config())
                          
    return keys2str(wandb_conf)

def get_autoinit_spec(obj):
    """
    gets autoinit args, together with the object class so that it can
    be instantiated again.
    """
    if not 'init_args' in dir(obj):
        raise ValueError(f"object {obj} was not initialized with autoinit")
    
    spec = {'init_args': obj.init_args,
            'class_name': obj.__class__.__name__,
            'class_module': obj.__class__.__module__}
    return spec        

def save_autoinit_spec(obj, file_path):
    """
    saves this instance init_args (set by autoinit) and class name
    """
    with open(file_path, "wb") as f:
        pickle.dump(get_autoinit_spec(obj), f)


def load_from_autoinit_spec(file_path):
    """
    loads spec saved by "save_conf" and restores an object instance
    with the class and init_args saved in file_path.
    """
    with open(file_path, "rb") as f:
        spec = pickle.load(f)        

    return restore_from_autoinit_spec(spec)

def restore_from_autoinit_spec(spec):
    """
    restores an object instance with the class and init_args in the spec.
    """
    cls = find_class(spec['class_name'])
    return cls(**spec['init_args'])


def keys2str(d):
    """
    converts recursively all keys which are not str, int, float, bool to string
    """
    r = {}
    for k,v in d.items():
        if isinstance(v, dict):
            v = keys2str(v)
        if type(k) in [str, int, float, bool, None]:
            r[k] = v
        else:
            r[str(k)] = v
    return r