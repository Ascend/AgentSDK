#!/usr/bin/env python3
# coding=utf-8
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
import asyncio
import pytest
from unittest.mock import patch, MagicMock
import ray
from ray.util.placement_group import PlacementGroup

from agentic_rl.base.resources.resources import (
    ResourceSet,
    create_placement_group_with_affinity,
    create_resource_set,
    _PG_CREATION_LOCK
)


class TestResourceSet:
    """Test the ResourceSet data model"""
    
    def test_resource_set_initialization(self):
        """Test basic initialization of ResourceSet"""
        bundles = [{"CPU": 1.0, "GPU": 2.0}]
        mock_pg = MagicMock(spec=PlacementGroup)
        
        resource_set = ResourceSet(info=bundles, ref=mock_pg)
        
        assert resource_set.info == bundles
        assert resource_set.ref == mock_pg
    
    def test_resource_set_without_ref(self):
        """Test ResourceSet initialization without reference"""
        bundles = [{"CPU": 1.0}]
        
        resource_set = ResourceSet(info=bundles, ref=None)
        
        assert resource_set.info == bundles
        assert resource_set.ref is None


class TestCreatePlacementGroupWithAffinity:
    """Test the create_placement_group_with_affinity function"""
    
    @pytest.mark.asyncio
    @patch("agentic_rl.base.resources.resources.ray._private.state.available_resources_per_node")
    @patch("agentic_rl.base.resources.resources.placement_group")
    async def test_create_pg_success(self, mock_placement_group, mock_available_resources):
        """Test successful placement group creation"""
        # Mock available resources
        mock_available_resources.return_value = {
            "node1": {"CPU": 4.0, "GPU": 2.0, "node_id": "node1"},
            "node2": {"CPU": 8.0, "GPU": 4.0, "node_id": "node2"}
        }
        
        # Mock placement group without spec to allow arbitrary attributes
        mock_pg = MagicMock()
        # Create a future that completes immediately
        future = asyncio.Future()
        future.set_result(None)
        mock_pg.ready.return_value = future
        # Mock id attribute
        mock_pg.id = MagicMock()
        mock_pg.id.hex.return_value = "mock_pg_id"
        mock_placement_group.return_value = mock_pg
        
        # Test data
        bundles = [{"CPU": 2.0, "GPU": 1.0}]
        timeout = 30.0
        
        # Call the function
        result = await create_placement_group_with_affinity(bundles, timeout)
        
        # Verify calls
        mock_available_resources.assert_called_once()
        mock_placement_group.assert_called_once_with(
            bundles=bundles, 
            strategy="STRICT_PACK", 
            _soft_target_node_id="node1"  # Should select node1 with least available resources
        )
        mock_pg.ready.assert_called_once()
        
        # Verify result
        assert result == mock_pg
    
    @pytest.mark.asyncio
    @patch("agentic_rl.base.resources.resources.ray._private.state.available_resources_per_node")
    @patch("agentic_rl.base.resources.resources.placement_group")
    async def test_create_pg_no_feasible_nodes(self, mock_placement_group, mock_available_resources):
        """Test placement group creation when no nodes have enough resources"""
        # Mock available resources (insufficient for requested bundles)
        mock_available_resources.return_value = {
            "node1": {"CPU": 1.0, "GPU": 0.0}
        }
        
        # Mock placement group without spec to allow arbitrary attributes
        mock_pg = MagicMock()
        # Create a future that completes immediately
        future = asyncio.Future()
        future.set_result(None)
        mock_pg.ready.return_value = future
        # Mock id attribute
        mock_pg.id = MagicMock()
        mock_pg.id.hex.return_value = "mock_pg_id"
        mock_placement_group.return_value = mock_pg
        
        # Test data
        bundles = [{"CPU": 2.0, "GPU": 1.0}]
        
        # Call the function
        result = await create_placement_group_with_affinity(bundles)
        
        # Verify placement group is created without soft target
        mock_placement_group.assert_called_once_with(
            bundles=bundles, 
            strategy="PACK"
        )
        
        assert result == mock_pg
    
    @pytest.mark.asyncio
    @patch("agentic_rl.base.resources.resources.ray._private.state.available_resources_per_node")
    @patch("agentic_rl.base.resources.resources.placement_group")
    async def test_create_pg_timeout(self, mock_placement_group, mock_available_resources):
        """Test placement group creation timeout"""
        # Mock available resources
        mock_available_resources.return_value = {
            "node1": {"CPU": 4.0, "GPU": 2.0}
        }
        
        # Mock placement group with slow ready()
        mock_pg = MagicMock()
        # Create a future that never completes (to simulate timeout)
        future = asyncio.Future()
        mock_pg.ready.return_value = future
        # Mock id attribute
        mock_pg.id = MagicMock()
        mock_pg.id.hex.return_value = "mock_pg_id"
        mock_placement_group.return_value = mock_pg
        
        # Test data with very short timeout
        bundles = [{"CPU": 2.0}]
        timeout = 0.1
        
        # Call the function and expect timeout
        with patch("agentic_rl.base.resources.resources.ray.util.placement_group") as mock_pg_module:
            # Setup mock for remove_placement_group
            mock_remove = MagicMock()
            mock_pg_module.remove_placement_group = mock_remove
            
            with pytest.raises(TimeoutError):
                await create_placement_group_with_affinity(bundles, timeout)
            
            # Verify cleanup
            mock_remove.assert_called_once_with(mock_pg)
    
    @pytest.mark.asyncio
    @patch("agentic_rl.base.resources.resources.ray._private.state.available_resources_per_node")
    @patch("agentic_rl.base.resources.resources.ray.available_resources")
    @patch("agentic_rl.base.resources.resources.placement_group")
    async def test_ray_api_compatibility(self, mock_placement_group, mock_available, mock_old_api):
        """Test compatibility when old Ray API is not available"""
        # Mock AttributeError for old API
        mock_old_api.side_effect = AttributeError()
        mock_available.return_value = {"CPU": 4.0}
        
        # Mock placement group without spec to allow arbitrary attributes
        mock_pg = MagicMock()
        # Create a future that completes immediately
        future = asyncio.Future()
        future.set_result(None)
        mock_pg.ready.return_value = future
        # Mock id attribute
        mock_pg.id = MagicMock()
        mock_pg.id.hex.return_value = "mock_pg_id"
        mock_placement_group.return_value = mock_pg
        
        # Test data
        bundles = [{"CPU": 2.0}]
        
        # Call the function
        result = await create_placement_group_with_affinity(bundles)
        
        # Verify it falls back to creating PG without soft target
        mock_placement_group.assert_called_once_with(
            bundles=bundles, 
            strategy="PACK"
        )
        
        assert result == mock_pg


class TestCreateResourceSet:
    """Test the create_resource_set function"""
    
    @pytest.mark.asyncio
    @patch("agentic_rl.base.resources.resources.create_placement_group_with_affinity")
    async def test_create_resource_set_success(self, mock_create_pg):
        """Test successful resource set creation"""
        # Mock placement group
        mock_pg = MagicMock()
        mock_create_pg.return_value = mock_pg
        
        # Test data
        bundles_info = [{"CPU": 1.0, "GPU": 1.0}]
        
        # Call the function with mocked ResourceSet to avoid validation
        with patch("agentic_rl.base.resources.resources.ResourceSet") as mock_resource_set:
            mock_resource_set_instance = MagicMock()
            mock_resource_set_instance.info = bundles_info
            mock_resource_set_instance.ref = mock_pg
            mock_resource_set.return_value = mock_resource_set_instance
            
            resource_set = await create_resource_set(bundles_info)
            
            # Verify calls
            mock_create_pg.assert_called_once_with(bundles_info)
            mock_resource_set.assert_called_once_with(info=bundles_info, ref=mock_pg)
            
            # Verify result
            assert resource_set == mock_resource_set_instance
            assert resource_set.info == bundles_info
            assert resource_set.ref == mock_pg
    
    @pytest.mark.asyncio
    @patch("agentic_rl.base.resources.resources.create_placement_group_with_affinity")
    @patch("agentic_rl.base.resources.resources._PG_CREATION_LOCK")
    async def test_create_resource_set_lock_usage(self, mock_lock, mock_create_pg):
        """Test that create_resource_set uses the module lock"""
        # Test data
        bundles_info = [{"CPU": 1.0}]
        
        # Setup mock lock with proper async context manager behavior
        mock_lock.__aenter__.return_value = None
        mock_lock.__aexit__.return_value = None
        
        # Setup mock to bypass ResourceSet validation
        mock_pg_instance = MagicMock()
        with patch("agentic_rl.base.resources.resources.ResourceSet") as mock_resource_set:
            mock_resource_set.return_value = mock_pg_instance
            await create_resource_set(bundles_info)
            
            # Verify lock is acquired and released
            mock_lock.__aenter__.assert_called_once()
            mock_lock.__aexit__.assert_called_once()
            # Verify ResourceSet was called with correct parameters
            mock_resource_set.assert_called_once_with(info=bundles_info, ref=mock_create_pg.return_value)
    
    @pytest.mark.asyncio
    @patch("agentic_rl.base.resources.resources.create_placement_group_with_affinity")
    async def test_create_resource_set_exception_propagation(self, mock_create_pg):
        """Test that exceptions from create_placement_group_with_affinity are propagated"""
        # Mock exception
        mock_create_pg.side_effect = RuntimeError("PG creation failed")
        
        # Test data
        bundles_info = [{"CPU": 1.0}]
        
        # Call the function and expect exception
        with pytest.raises(RuntimeError, match="PG creation failed"):
            await create_resource_set(bundles_info)