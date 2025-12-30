#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import unittest
import tempfile
import os
import sys
import shutil
from unittest.mock import patch, MagicMock
from abc import ABC, abstractmethod

from agentic_rl.base.utils.class_loader import load_subclasses_from_file


class BaseClass:
    pass


class AbstractBase(ABC):
    @abstractmethod
    def abstract_method(self):
        pass


current_module = sys.modules[__name__]
setattr(current_module, 'BaseClass', BaseClass)
setattr(current_module, 'AbstractBase', AbstractBase)


class TestLoadSubclassesFromFile(unittest.TestCase):
    """Comprehensive test suite for load_subclasses_from_file function."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_files = []

    def tearDown(self):
        """Clean up temporary test files and directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_file(self, filename, content):
        """
        Helper to create temporary Python files with given content.
        
        Args:
            filename: Name of the file to create
            content: Python code content for the file
            
        Returns:
            Full path to the created file
        """
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        # Set proper permissions (640 for files)
        os.chmod(file_path, 0o640)
        self.test_files.append(file_path)
        return file_path

    # ========== Success Cases ==========

    def test_load_single_subclass_success(self):
        """Test loading a valid subclass from a file."""
        content = '''
from test_class_loader import BaseClass

class SubClass(BaseClass):
    """Single subclass."""
    pass
'''
        file_path = self.create_test_file('valid_subclass.py', content)

        # Load the subclass
        result = load_subclasses_from_file(file_path, BaseClass)

        # Verify result
        self.assertIsNotNone(result)
        self.assertTrue(isinstance(result, type))
        self.assertEqual(result.__name__, 'SubClass')
        self.assertTrue(issubclass(result, BaseClass))

    def test_load_subclass_with_abstract_base_class(self):
        """Test loading subclass with abstract base class (ABC)."""
        content = '''
from abc import ABC, abstractmethod
from test_class_loader import AbstractBase


class ConcreteSubclass(AbstractBase):
    """Concrete implementation."""
    
    def abstract_method(self):
        return "implemented"
'''
        file_path = self.create_test_file('abstract_test.py', content)
        
        result = load_subclasses_from_file(file_path, AbstractBase)
        
        self.assertEqual(result.__name__, 'ConcreteSubclass')
        self.assertTrue(issubclass(result, AbstractBase))

    def test_load_subclass_inheritance_chain_single_direct(self):
        """Test loading when only one direct subclass exists."""
        content = '''
class DirectSubclass:
    """Only subclass that will match."""
    pass
'''
        file_path = self.create_test_file('single_class.py', content)

        content = '''
from test_class_loader import BaseClass

class DirectSubclass(BaseClass):
    """Only subclass."""
    pass
'''
        file_path = self.create_test_file('single_direct.py', content)
        
        result = load_subclasses_from_file(file_path, BaseClass)
        self.assertEqual(result.__name__, 'DirectSubclass')

    def test_load_subclass_with_multiple_inheritance(self):
        """Test subclass inheriting from multiple classes including base_class."""
        content = '''
from test_class_loader import BaseClass

class Mixin:
    """Mixin class."""
    pass

class MultiInheritSubclass(BaseClass, Mixin):
    """Subclass with multiple inheritance."""
    pass
'''
        file_path = self.create_test_file('multi_inherit.py', content)
        
        result = load_subclasses_from_file(file_path, BaseClass)
        self.assertEqual(result.__name__, 'MultiInheritSubclass')
        self.assertTrue(issubclass(result, BaseClass))

    # ========== Parameter Validation Tests ==========

    def test_invalid_base_class_string(self):
        """Test passing a string instead of a class raises ValueError."""
        file_path = self.create_test_file('dummy.py', 'pass')
        
        with self.assertRaises(ValueError) as context:
            load_subclasses_from_file(file_path, "not_a_class")
        self.assertIn("base_class must be a class", str(context.exception))

    def test_invalid_base_class_integer(self):
        """Test passing an integer instead of a class raises ValueError."""
        file_path = self.create_test_file('dummy.py', 'pass')
        
        with self.assertRaises(ValueError) as context:
            load_subclasses_from_file(file_path, 123)
        self.assertIn("base_class must be a class", str(context.exception))

    def test_invalid_base_class_instance(self):
        """Test passing a class instance instead of class raises ValueError."""
        file_path = self.create_test_file('dummy.py', 'pass')
        
        class MyClass:
            pass
        
        instance = MyClass()
        
        with self.assertRaises(ValueError) as context:
            load_subclasses_from_file(file_path, instance)
        self.assertIn("base_class must be a class", str(context.exception))

    def test_invalid_base_class_none(self):
        """Test passing None as base_class raises ValueError."""
        file_path = self.create_test_file('dummy.py', 'pass')
        
        with self.assertRaises(ValueError) as context:
            load_subclasses_from_file(file_path, None)
        self.assertIn("base_class must be a class", str(context.exception))

    # ========== File Path Validation Tests ==========

    def test_non_existent_file(self):
        """Test loading from non-existent file raises ValueError."""
        non_existent_path = os.path.join(self.temp_dir, 'does_not_exist.py')
        
        with self.assertRaises(ValueError) as context:
            load_subclasses_from_file(non_existent_path, BaseClass)
        self.assertIn("path is not existed", str(context.exception))

    @patch('agentic_rl.base.utils.file_utils.FileCheck.check_data_path_is_valid')
    def test_file_path_too_long(self, mock_check):
        """Test file path that's too long raises ValueError."""
        mock_check.side_effect = ValueError("Input path is too long, it's length must be less than 1024.")
        
        long_path = "a" * 1025 + ".py"
        
        with self.assertRaises(ValueError) as context:
            load_subclasses_from_file(long_path, BaseClass)
        self.assertIn("too long", str(context.exception))

    @patch('agentic_rl.base.utils.file_utils.FileCheck.check_data_path_is_valid')
    def test_file_path_with_illegal_characters(self, mock_check):
        """Test file path with illegal characters raises ValueError."""
        mock_check.side_effect = ValueError("There are illegal characters in path, it must be in [a-z A-Z 0-9 . _ -].")
        
        illegal_path = "/path/with/illegal@char.py"
        
        with self.assertRaises(ValueError) as context:
            load_subclasses_from_file(illegal_path, BaseClass)
        self.assertIn("illegal characters", str(context.exception))

    @patch('agentic_rl.base.utils.file_utils.FileCheck.check_data_path_is_valid')
    def test_file_path_is_symlink(self, mock_check):
        """Test file path that's a symlink raises ValueError."""
        mock_check.side_effect = ValueError("Path is link, it's not supported.")
        
        symlink_path = "/path/to/symlink.py"
        
        with self.assertRaises(ValueError) as context:
            load_subclasses_from_file(symlink_path, BaseClass)
        self.assertIn("link", str(context.exception))

    # ========== Module Name Validation Tests ==========

    def test_invalid_module_name_starts_with_number(self):
        """Test file name that starts with number (invalid Python identifier)."""
        content = '''
from test_class_loader import BaseClass

class SubClass(BaseClass):
    pass
'''
        file_path = self.create_test_file('123invalid.py', content)
        
        with self.assertRaises(ValueError) as context:
            load_subclasses_from_file(file_path, BaseClass)
        self.assertIn("Invalid module name", str(context.exception))
        self.assertIn("must be a valid Python identifier", str(context.exception))

    def test_invalid_module_name_with_hyphen(self):
        """Test file name with hyphen (invalid Python identifier)."""
        content = '''
from test_class_loader import BaseClass

class SubClass(BaseClass):
    pass
'''
        file_path = self.create_test_file('class-name.py', content)
        
        with self.assertRaises(ValueError) as context:
            load_subclasses_from_file(file_path, BaseClass)
        self.assertIn("Invalid module name", str(context.exception))
        self.assertIn("must be a valid Python identifier", str(context.exception))

    def test_invalid_module_name_with_space(self):
        """Test file name with space (invalid Python identifier)."""
        content = '''
from test_class_loader import BaseClass

class SubClass(BaseClass):
    pass
'''
        file_path = self.create_test_file('my module.py', content)

        with self.assertRaises(ValueError) as context:
            load_subclasses_from_file(file_path, BaseClass)
        self.assertIn("illegal characters in path", str(context.exception))

    # ========== Module Loading Failures ==========

    @patch('importlib.util.spec_from_file_location')
    def test_spec_from_file_location_returns_none(self, mock_spec_from_file):
        """Test when spec_from_file_location returns None."""
        mock_spec_from_file.return_value = None
        
        file_path = self.create_test_file('valid.py', 'pass')
        
        with self.assertRaises(ImportError) as context:
            load_subclasses_from_file(file_path, BaseClass)
        self.assertIn("Unable to load module from", str(context.exception))

    @patch('importlib.util.spec_from_file_location')
    def test_spec_loader_is_none(self, mock_spec_from_file):
        """Test when spec.loader is None."""
        mock_spec = MagicMock()
        mock_spec.loader = None
        mock_spec_from_file.return_value = mock_spec
        
        file_path = self.create_test_file('valid.py', 'pass')
        
        with self.assertRaises(ImportError) as context:
            load_subclasses_from_file(file_path, BaseClass)
        self.assertIn("Unable to load module from", str(context.exception))

    def test_module_with_syntax_error(self):
        """Test loading module with syntax errors."""
        content = '''
from test_class_loader import BaseClass

class SubClass(BaseClass)  # Missing colon
    pass
'''
        file_path = self.create_test_file('syntax_error.py', content)
        
        with self.assertRaises(ImportError) as context:
            load_subclasses_from_file(file_path, BaseClass)
        self.assertIn("Unexpected error occurred during executing module", str(context.exception))

    def test_module_with_runtime_error(self):
        """Test loading module that raises exception during execution."""
        content = '''
# This will raise an error during module execution
raise RuntimeError("Module initialization failed")

from test_class_loader import BaseClass

class SubClass(BaseClass):
    pass
'''
        file_path = self.create_test_file('runtime_error.py', content)

        with self.assertRaises(ImportError) as context:
            load_subclasses_from_file(file_path, BaseClass)
        self.assertIn("Failed to execute module", str(context.exception))
        self.assertIn("Module initialization failed", str(context.exception))

    def test_module_with_import_error(self):
        """Test loading module with missing import."""
        content = '''
import non_existent_module

from test_class_loader import BaseClass

class SubClass(BaseClass):
    pass
'''
        file_path = self.create_test_file('import_error.py', content)
        
        with self.assertRaises(ImportError) as context:
            load_subclasses_from_file(file_path, BaseClass)
        self.assertIn("Unexpected error occurred during executing module", str(context.exception))

    # ========== Subclass Discovery Tests ==========

    def test_no_subclasses_found(self):
        """Test module with classes but none inherit from base_class."""
        content = '''
class UnrelatedClass:
    """This doesn't inherit from BaseClass."""
    pass

class AnotherUnrelatedClass:
    """Neither does this."""
    pass
'''
        file_path = self.create_test_file('no_subclasses.py', content)

        with self.assertRaises(ImportError) as context:
            load_subclasses_from_file(file_path, BaseClass)
        self.assertIn("found no class inherited from", str(context.exception))
        self.assertIn("BaseClass", str(context.exception))

    def test_multiple_subclasses_found(self):
        """Test module with multiple subclasses raises ImportError."""
        content = '''
from test_class_loader import BaseClass

class SubClass1(BaseClass):
    """First subclass."""
    pass

class SubClass2(BaseClass):
    """Second subclass."""
    pass
'''
        file_path = self.create_test_file('multiple_subclasses.py', content)

        with self.assertRaises(ImportError) as context:
            load_subclasses_from_file(file_path, BaseClass)
        self.assertIn("multiply classes inherited", str(context.exception))
        self.assertIn("ensure there is just one subclass", str(context.exception))

    def test_base_class_itself_excluded(self):
        """Test that base class defined in module is not counted as subclass."""
        content = '''
from test_class_loader import BaseClass

class SubClass(BaseClass):
    """The actual subclass."""
    pass
'''
        file_path = self.create_test_file('with_base_class.py', content)

        result = load_subclasses_from_file(file_path, BaseClass)

        # Should only find SubClass, not BaseClass itself
        self.assertEqual(result.__name__, 'SubClass')
        self.assertNotEqual(result.__name__, 'BaseClass')

    def test_empty_module(self):
        """Test module with no classes at all."""
        content = '''
# Empty module with just comments
x = 1
y = 2

def some_function():
    pass
'''
        file_path = self.create_test_file('empty.py', content)

        with self.assertRaises(ImportError) as context:
            load_subclasses_from_file(file_path, BaseClass)
        self.assertIn("found no class inherited from", str(context.exception))

    # ========== Edge Cases ==========

    def test_subclass_with_methods_and_attributes(self):
        """Test subclass with various methods and attributes."""
        content = '''
from test_class_loader import BaseClass

class ComplexSubclass(BaseClass):
    """Subclass with methods and attributes."""
    
    class_attr = "test"
    
    def __init__(self):
        self.instance_attr = "value"
    
    def method(self):
        return "result"
    
    @staticmethod
    def static_method():
        return "static"
    
    @classmethod
    def class_method(cls):
        return "class"
'''
        file_path = self.create_test_file('complex_subclass.py', content)

        result = load_subclasses_from_file(file_path, BaseClass)
        
        self.assertEqual(result.__name__, 'ComplexSubclass')
        self.assertTrue(issubclass(result, BaseClass))
        self.assertTrue(hasattr(result, 'class_attr'))
        self.assertTrue(hasattr(result, 'method'))
        self.assertTrue(hasattr(result, 'static_method'))
        self.assertTrue(hasattr(result, 'class_method'))

    def test_subclass_with_decorators(self):
        """Test subclass with class decorators."""
        content = '''
from test_class_loader import BaseClass

def decorator(cls):
    cls.decorated = True
    return cls

@decorator
class DecoratedSubclass(BaseClass):
    """Decorated subclass."""
    pass
'''
        file_path = self.create_test_file('decorated.py', content)

        result = load_subclasses_from_file(file_path, BaseClass)
        
        self.assertEqual(result.__name__, 'DecoratedSubclass')
        self.assertTrue(issubclass(result, BaseClass))
        self.assertTrue(hasattr(result, 'decorated'))
        self.assertTrue(result.decorated)

    def test_file_with_valid_module_name_edge_cases(self):
        """Test valid module names that are edge cases but acceptable."""
        # Underscore is valid
        content = '''
from test_class_loader import BaseClass

class SubClass(BaseClass):
    pass
'''
        file_path = self.create_test_file('_valid_module.py', content)
        
        result = load_subclasses_from_file(file_path, BaseClass)
        self.assertEqual(result.__name__, 'SubClass')

    def test_resolve_relative_path(self):
        """Test that relative paths are resolved correctly."""
        content = '''
from test_class_loader import BaseClass

class SubClass(BaseClass):
    pass
'''
        filename = 'relative_test.py'
        file_path = self.create_test_file(filename, content)
        
        result = load_subclasses_from_file(file_path, BaseClass)
        self.assertEqual(result.__name__, 'SubClass')


if __name__ == '__main__':
    unittest.main()
