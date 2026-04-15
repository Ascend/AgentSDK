#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# 
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# 
#          http://license.coscl.org.cn/MulanPSL2
# 
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


import os
import sys
import unittest
import asyncio
from unittest.mock import patch, MagicMock


# Create mock objects
mock_ray = MagicMock()

# Mock required dependency modules before importing the module under test
with patch.dict('sys.modules', {
    'ray': mock_ray
}):
    # Now we can safely import the class under test
    from agentic_rl.trainer.train_adapter.verl.full_async.message_queue import MessageQueueDMClient

class TestMessageQueueDMClient(unittest.TestCase):
    def setUp(self):
        # Create mock queue_actor
        self.mock_queue_actor = MagicMock()
        
        # Create test instance
        self.client = MessageQueueDMClient(queue_actor=self.mock_queue_actor)
    
    def test_init(self):
        # Verify initialization
        self.assertEqual(self.client.queue_actor, self.mock_queue_actor)
    
    def test_put_sample_sync(self):
        # Prepare test data
        sample = {'data': 'test_sample'}
        param_version = 1
        
        # Call the method
        result = self.client.put_sample_sync(sample, param_version)
        
        # Since the original method has no return value, verify result is None
        self.assertIsNone(result)
    
    def test_get_sample_sync(self):
        # Call the method
        result = self.client.get_sample_sync()
        
        # Since the original method has no return value, verify result is None
        self.assertIsNone(result)
    
    def test_get_validate_sync(self):
        # Call the method
        result = self.client.get_validate_sync()
        
        # Since the original method has no return value, verify result is None
        self.assertIsNone(result)
    
    def test_get_statistics_sync(self):
        # Call the method
        result = self.client.get_statistics_sync()
        
        # Since the original method has no return value, verify result is None
        self.assertIsNone(result)
    
    def test_update_param_version_sync(self):
        # Prepare test data
        version = 1
        
        # Call the method
        result = self.client.update_param_version_sync(version)
        
        # Since the original method has no return value, verify result is None
        self.assertIsNone(result)
    
    async def test_put_sample(self):
        # Prepare test data
        sample = {'data': 'test_sample'}
        param_version = 1
        
        # Call the async method
        result = await self.client.put_sample(sample, param_version)
        
        # Since the original method has no return value, verify result is None
        self.assertIsNone(result)
    
    async def test_put_validate(self):
        # Prepare test data
        data = {'validate': 'test_data'}
        
        # Call the async method
        result = await self.client.put_validate(data)
        
        # Since the original method has no return value, verify result is None
        self.assertIsNone(result)
    
    async def test_get_sample(self):
        # Call the async method
        result = await self.client.get_sample()
        
        # Since the original method has no return value, verify result is None
        self.assertIsNone(result)
    
    async def test_get_queue_size(self):
        # Call the async method
        result = await self.client.get_queue_size()
        
        # Since the original method has no return value, verify result is None
        self.assertIsNone(result)
    
    async def test_get_statistics(self):
        # Call the async method
        result = await self.client.get_statistics()
        
        # Since the original method has no return value, verify result is None
        self.assertIsNone(result)
    
    async def test_clear_queue(self):
        # Call the async method
        await self.client.clear_queue()
    
    async def test_shutdown(self):
        # Call the async method
        await self.client.shutdown()
    
    async def test_get_memory_usage(self):
        # Call the async method
        result = await self.client.get_memory_usage()
        
        # Since the original method has no return value, verify result is None
        self.assertIsNone(result)
    
    # Test case without queue_actor
    def test_init_without_queue_actor(self):
        # Create test instance (without providing queue_actor)
        client = MessageQueueDMClient()
        
        # Verify queue_actor is None
        self.assertIsNone(client.queue_actor)

# Helper function to run async tests
def async_test(coro):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper

# 为所有异步测试方法应用async_test装饰器
for name in dir(TestMessageQueueDMClient):
    if name.startswith('test_') and asyncio.iscoroutinefunction(getattr(TestMessageQueueDMClient, name)):
        setattr(TestMessageQueueDMClient, name, async_test(getattr(TestMessageQueueDMClient, name)))

if __name__ == '__main__':
    unittest.main()