from dataclasses import dataclass
import sys
from typing import get_args
import types


class CommandLineFlag:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        return str(self.__dict__)
    

@dataclass
class CommandLineArgument:
    def __init__(self):
        self.__defined_arguments = {}
    
    def define(self, name: str, default=None, arg_type=None, *, help=""):
        """
            Defines an argument that is passed via the command line.
            For example:
            ```
                define("EPOCHS", default=10, arg_type=int)
            ```
            Usage:
            ```
                $ script.py --EPOCHS=20
            ```

            Arguments:
                name <str>: Name of argument (when using, add prefix '--')

                default <arg_type>: Default value of argument, must match arg_type

                arg_type <type>: Type of argument

                help: See instructions on using argument
                :param help:
                :param arg_type:
                :param name:
                :param default:

            """
        
        if not isinstance(name, str):
            raise TypeError(f"Argument 'name' is expected to be of type 'str' but got '{type(name)}'!")
        
        if not isinstance(default, arg_type):
            if arg_type is float and isinstance(default, int):
                default = float(default)
            else:
                raise TypeError(
                    f"Argument '{name}' with default value {default} is expected to be of type '{arg_type}' but got '{type(default)}'!")
        
        if name == "":
            raise ValueError(f"Argument 'name' cannot be empty!")
        
        if '-' == name[0]:
            raise ValueError(f"Argument 'name' cannot contain leading dash: {name}!")
        
        if '=' in name:
            raise ValueError(f"Argument 'name' cannot contain '=': {name}!")
        
        if str in self.__defined_arguments:
            raise ValueError(f"Argument 'name' was previously defined: {name}!")
        
        if isinstance(arg_type, type):
            if type is float and isinstance(default, int):
                default = float(type)
            self.__defined_arguments[name] = {"type": arg_type, "union": False, "value": default, "help": help}
        
        elif isinstance(arg_type, types.UnionType):
            types_list = list(get_args(arg_type))
            if int in types_list and float in types_list:
                print("Use float instead of 'int | float'")
                types_list.remove(int)
            for tp in types_list:
                if tp not in [int, float, str, bool]:
                    raise ValueError(f"Type '{tp}' currently has not supported!")
            
            self.__defined_arguments[name] = {"type": types_list, "union": True, "value": default, "help": help}
        
        else:
            raise ValueError(f"Argument 'arg_type' is not a variable type: {arg_type}!")
    
    def count_dashes(self, string):
        """
            Đếm số lượng ký tự '-' liên tiếp từ đầu chuỗi

            Args:
                string: Chuỗi cần đếm

            Returns:
                Số lượng ký tự '-' liên tiếp từ đầu chuỗi
            """
        
        count = 0
        for i in string:
            if i == '-':
                count += 1
            else:
                break
        return count
    
    def __match(self):
        argument_strings = sys.argv[1:]
        
        for arg_str in argument_strings:
            delimiter_index = arg_str.index('=')
            name = arg_str[:delimiter_index]
            value = arg_str[delimiter_index + 1:]
            num_prefix_dash = self.count_dashes(name)
            if num_prefix_dash == 2:
                name = name[2:]
                if name in self.__defined_arguments:
                    arg_info = self.__defined_arguments[name]
                    if arg_info["union"]:
                        arg_types = arg_info["type"]
                        passed = False
                        if int in arg_types:
                            try:
                                arg_info["value"] = int(value)
                                passed = True
                            except ValueError:
                                pass
                        if float in arg_types:
                            try:
                                arg_info["value"] = float(value)
                                passed = True
                            except ValueError:
                                pass
                        if bool in arg_types:
                            if value == "True" or value == "true":
                                arg_info["value"] = True
                                passed = True
                            elif value == "False" or value == "false":
                                arg_info["value"] = False
                                passed = True
                        if not passed:
                            if str in arg_types:
                                arg_info["value"] = value
                            else:
                                raise ValueError(
                                    f"Value {value} of argument {name} does not match type {arg_types}!")
                    else:
                        arg_type = arg_info["type"]
                        try:
                            arg_info["value"] = arg_type(value)
                            if arg_type is int:
                                arg_info["value"] = int(value)
                            elif arg_type is float:
                                arg_info["value"] = float(value)
                            elif arg_type is bool:
                                if value == "True" or value == "true":
                                    arg_info["value"] = True
                                elif value == "False" or value == "false":
                                    arg_info["value"] = False
                            
                            elif arg_type is str:
                                arg_info["value"] = value
                            else:
                                raise ValueError(f"Argument {name} does not support type {arg_type}!")
                        except ValueError:
                            raise ValueError(f"Value {value} of argument {name} does not match type {arg_type}!")
                else:
                    raise ValueError(f"Argument {name} does not exist!")
            else:
                raise ValueError("Invalid flag prefix!")
    
    def parse(self, fetch: bool = True):
        if fetch:
            self.__match()
        return CommandLineFlag(**{k: v["value"] for k, v in self.__defined_arguments.items()})
    