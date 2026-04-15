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
import typing
from typing import List, Dict

import ray
from pydantic import BaseModel, ConfigDict
from ray.util.placement_group import PlacementGroup
from ray.util.placement_group import placement_group

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


class ResourceSet(BaseModel):
    info: List[Dict[str, float]]
    ref: typing.Optional[PlacementGroup]

    model_config = ConfigDict(arbitrary_types_allowed=True)


async def create_placement_group_with_affinity(
        bundles: typing.List[typing.Dict[str, float]],
        timeout_seconds: float = 300.0
) -> PlacementGroup:
    """
    Asynchronously create a Placement Group, prioritizing it on nodes with high resource occupancy rates,
    and supporting custom resources.

    Args:
        bundles: Resource package list, defining the resources required by PG.
        timeout_seconds: The timeout period (in seconds) for waiting for PG to be ready.

    Returns:
        Create a successfully created PlacementGroup instance.
    """

    def get_candidate_nodes() -> typing.List[typing.Dict[str, typing.Any]]:
        """Obtain active nodes and their available resources (synchronous operation)"""
        try:
            # This is a synchronous call to the Ray State API.
            # Note: ray._private.state is an internal API of Ray and may change in future versions.
            # This API is inherently synchronous and does not release the GIL when called within a coroutine.
            # However, since it is a quick status query, its impact is usually minimal.
            available_per_node = ray._private.state.available_resources_per_node()
        except AttributeError:
            # Compatibility handling: If the Ray versions are different,
            # alternative methods may be required to obtain the resources.
            logger.debug("Warning: ray._private.state.available_resources_per_node() not found. "
                         "Falling back to ray.available_resources().")
            # Return an empty list and cause the scheduling to roll back.
            return []

        candidates = []
        # Traverse all nodes and their available resources
        for node_id, resources in available_per_node.items():
            # Calculate the total available resources, excluding internal resources (such as bundle_group and node_*)
            total_avail = sum(
                v for r, v in resources.items()
                if not r.startswith("bundle_group") and not r.startswith("node")
            )
            candidates.append({
                "node_id": node_id,
                # The smaller the avail_sum is, the fuller the node is and the higher its priority is.
                "avail_sum": total_avail,
                "available": resources
            })
        return candidates

    def filter_feasible_nodes(
            nodes: typing.List[typing.Dict[str, typing.Any]]
    ) -> typing.List[typing.Dict[str, typing.Any]]:
        """Filter out the nodes that can accommodate the current PG (synchronous operation)"""
        feasible = []
        # Sum up the resource requirements of all bundles
        total_needed: typing.Dict[str, float] = {}
        for bundle in bundles:
            for r, v in bundle.items():
                total_needed[r] = total_needed.get(r, 0) + v

        for n in nodes:
            avail = n["available"]
            # "all()" ensures that all the required resources r satisfy the condition avail[r] >= v
            if all(avail.get(r, 0) >= v for r, v in total_needed.items()):
                feasible.append(n)
        return feasible

    def select_soft_target_node(feasible_nodes: typing.List[typing.Dict[str, typing.Any]]) -> typing.Optional[str]:
        """
        Select the node ID with the least available resources,
        i.e., the highest occupancy rate (for the synchronization operation)
        """
        if not feasible_nodes:
            return None
        # Sort by 'avail_sum' in ascending order: the smaller the 'avail_sum',
        # the fuller the node, and the higher the priority.
        feasible_nodes.sort(key=lambda n: n["avail_sum"])
        return feasible_nodes[0]["node_id"]

    async def create_pg_on_node(
            node_id: typing.Optional[str] = None
    ) -> PlacementGroup:
        """Asynchronously create PG. If node_id is provided, use _soft_target_node_id instead."""
        kwargs = {"bundles": bundles, "strategy": "PACK"}

        if node_id:
            # Implement soft affinity by using Ray's private parameter _soft_target_node_id
            kwargs["strategy"] = "STRICT_PACK"
            kwargs["_soft_target_node_id"] = node_id
            logger.debug(f"Attempting to soft target node: {node_id}")
        else:
            logger.debug("No feasible node found. Creating placement group without soft affinity.")

        # 1. Create PG handle
        pg = placement_group(**kwargs)

        # 2. Asynchronous waiting for PG to be ready (the crucial timeout mechanism)
        logger.debug(f"Waiting for PG to be ready (Timeout: {timeout_seconds}s)...")
        try:
            # Wrap pg.ready() with asyncio.wait_for
            await asyncio.wait_for(pg.ready(), timeout=timeout_seconds)

        except asyncio.TimeoutError:
            # Perform cleanup and throw a custom exception during timeout.
            error_msg = (
                f"Placement Group creation timed out after {timeout_seconds} seconds. "
                f"PG ID: {pg.id.hex()}, Bundles: {bundles}"
            )
            logger.debug(f"Error: {error_msg}")

            # Attempt to remove PG to free up the resources that might be occupied.
            try:
                ray.util.placement_group.remove_placement_group(pg)
                logger.debug(f"Cleaned up timed out Placement Group: {pg.id.hex()}")
            except Exception as e:
                logger.debug(f"Warning: Failed to clean up PG {pg.id.hex()}: {e}")

            raise TimeoutError(error_msg)

        logger.debug(f"Placement Group created and ready. ID: {pg.id.hex()}")
        return pg

    logger.debug("Fetching candidate nodes...")
    out_candidate_nodes = get_candidate_nodes()

    logger.debug(f"Found {len(out_candidate_nodes)} candidate nodes.")
    out_feasible_nodes = filter_feasible_nodes(out_candidate_nodes)

    logger.debug(f"Found {len(out_feasible_nodes)} feasible nodes.")
    out_soft_target_node_id = select_soft_target_node(out_feasible_nodes)

    return await create_pg_on_node(out_soft_target_node_id)


# ----------------- Module-level Lock -----------------
# Used to ensure that the create_pg_with_affinity method is executed serially as a whole,
# avoiding concurrent modifications or query competition.
_PG_CREATION_LOCK = asyncio.Lock()


async def create_resource_set(bundles_info: List[Dict[str, float]]) -> ResourceSet:
    async with _PG_CREATION_LOCK:
        pg_ref = await create_placement_group_with_affinity(bundles_info)
    return ResourceSet(info=bundles_info, ref=pg_ref)
