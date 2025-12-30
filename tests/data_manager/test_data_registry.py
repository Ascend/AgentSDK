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
from unittest.mock import patch, MagicMock

from agentic_rl.data_manager.data_registry import DataManagerRegistry, data_manager_class


class TestClass:
    pass


class TestDataManagerRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = DataManagerRegistry()
        self.test_class = TestClass
        self.registry._registry = {'backend1': 'class1', 'backend2': 'class2'}

    def test_register_with_invalid_train_backend(self):
        with self.assertRaises(ValueError):
            self.registry.register(None, MagicMock)

    def test_register_with_non_string_train_backend(self):
        with self.assertRaises(ValueError):
            self.registry.register(123, MagicMock)

    def test_register_with_non_class_cls(self):
        with self.assertRaises(TypeError):
            self.registry.register('valid_train_backend', 'not_a_class')

    def test_register_successfully(self):
        self.registry.register('valid_backend', self.test_class)
        self.assertEqual(self.registry._registry.get('valid_backend'), self.test_class)
        self.assertIn('valid_backend', self.registry._registry)

    def test_get_class_with_valid_backend(self):
        self.assertEqual(self.registry.get_class('backend1'), 'class1')
        self.assertEqual(self.registry.get_class('backend2'), 'class2')

    def test_get_class_with_invalid_backend(self):
        with self.assertRaises(KeyError):
            self.registry.get_class('backend3')

    def test_get_class_with_non_string_backend(self):
        with self.assertRaises(ValueError):
            self.registry.get_class(123)
        with self.assertRaises(ValueError):
            self.registry.get_class(None)

    def test_get_class_with_empty_string_backend(self):
        with self.assertRaises(ValueError):
            self.registry.get_class('')

    @patch('agentic_rl.data_manager.data_registry.DataManagerRegistry.get_class')
    def test_get_class_with_valid_train_backend(self, mock_get_class):
        mock_get_class.return_value = 'SomeClass'
        self.assertEqual(data_manager_class('some_train_backend'), 'SomeClass')

    @patch('agentic_rl.data_manager.data_registry.DataManagerRegistry.get_class')
    def test_get_class_with_none_train_backend(self, mock_get_class):
        with self.assertRaises(ValueError) as context:
            data_manager_class(None)
        self.assertIn('train_backend must be a non-empty string', str(context.exception))

    @patch('agentic_rl.data_manager.data_registry.DataManagerRegistry.get_class')
    def test_get_class_with_empty_train_backend(self, mock_get_class):
        with self.assertRaises(ValueError) as context:
            data_manager_class('')
        self.assertIn('train_backend must be a non-empty string', str(context.exception))

    @patch('agentic_rl.data_manager.data_registry.DataManagerRegistry.get_class')
    def test_get_class_with_non_string_train_backend(self, mock_get_class):
        with self.assertRaises(ValueError) as context:
            data_manager_class(123)
        self.assertIn('train_backend must be a non-empty string', str(context.exception))

    @patch('agentic_rl.data_manager.data_registry.DataManagerRegistry.get_class')
    def test_get_class_with_no_class_found(self, mock_get_class):
        mock_get_class.return_value = None
        with self.assertRaises(ValueError) as context:
            data_manager_class('some_train_backend')
        self.assertIn('No data manager class found for train_backend', str(context.exception))


if __name__ == '__main__':
    unittest.main()