/**
 * Task Graph Service
 *
 * Provides task dependency graph analysis including:
 * - Build task dependency graphs
 * - Find executable tasks (no pending dependencies)
 * - Detect circular dependencies
 * - Compute critical path
 */

import type { Task } from "@subagent-coordinator/types";
import type { TaskStoreService } from "./task_store";

export interface TaskNode {
  task: Task;
  dependencies: string[];
  dependents: string[];
}

export interface TaskGraph {
  nodes: Map<string, TaskNode>;
  rootTaskId: string | null;
}

export interface CriticalPathResult {
  path: string[];
  totalEstimatedDuration: number;
}

export interface TaskGraphState {
  nodes: Map<string, TaskNode>;
}

export interface TaskGraphService {
  buildGraph(rootTaskId?: string): TaskGraph;
  getExecutableTasks(): Task[];
  getTaskPath(taskId: string): string[];
  detectCycles(): string[][];
  getCriticalPath(rootTaskId: string): CriticalPathResult;
  topologicalSort(): string[];
  getParallelGroups(rootTaskId: string): string[][];
}

export function createTaskGraph(
  taskStore: TaskStoreService
): TaskGraphService {
  /**
   * Build a task graph from all tasks or a specific root task
   */
  function buildGraph(rootTaskId?: string): TaskGraph {
    const nodes = new Map<string, TaskNode>();
    const allTasks = taskStore.list();

    // Build nodes
    for (const task of allTasks) {
      const deps = taskStore.getDependencies(task.id);
      nodes.set(task.id, {
        task,
        dependencies: deps,
        dependents: []
      });
    }

    // Build reverse dependencies (dependents)
    for (const [taskId, node] of nodes.entries()) {
      for (const depId of node.dependencies) {
        const depNode = nodes.get(depId);
        if (depNode) {
          depNode.dependents.push(taskId);
        }
      }
    }

    // Determine root task
    let rootId = rootTaskId || null;
    if (!rootId && nodes.size > 0) {
      // Find tasks with no dependencies
      const roots = Array.from(nodes.values()).filter(n => n.dependencies.length === 0);
      rootId = roots[0]?.task.id || null;
    }

    return { nodes, rootTaskId: rootId };
  }

  /**
   * Get all tasks that can be executed now (no pending dependencies)
   */
  function getExecutableTasks(): Task[] {
    const allTasks = taskStore.list();
    const blockedByStatus = new Set<string>();

    // Find tasks that are blocked by in-progress parent tasks
    for (const task of allTasks) {
      const status = (task as any).status;
      if (status === "wip") {
        // Task is in progress, block its subtasks
        const subtasks = taskStore.getSubtasks(task.id);
        for (const subtask of subtasks) {
          blockedByStatus.add(subtask.id);
        }
      }
    }

    // Filter tasks with no unmet dependencies
    return allTasks.filter(task => {
      const taskId = task.id;

      // Skip blocked tasks
      if (blockedByStatus.has(taskId)) {
        return false;
      }

      const status = (task as any).status;

      // Skip completed tasks
      if (status === "done" || status === "skipped") {
        return false;
      }

      const deps = taskStore.getDependencies(taskId);
      for (const depId of deps) {
        const depTask = taskStore.get(depId);
        if (!depTask) continue; // Dependency task doesn't exist, skip
        const depStatus = (depTask as any).status;
        if (depStatus !== "done" && depStatus !== "skipped") {
          return false; // Dependency not satisfied
        }
      }

      return true;
    });
  }

  /**
   * Get the path from root to a specific task
   */
  function getTaskPath(taskId: string): string[] {
    const path: string[] = [];
    const visited = new Set<string>();

    function traverse(id: string): boolean {
      if (visited.has(id)) return true; // Cycle detected
      visited.add(id);
      path.push(id);

      // Find parent
      const parent = taskStore.getParent(id);
      if (parent) {
        return traverse(parent.id);
      }

      return false;
    }

    traverse(taskId);
    return path.reverse();
  }

  /**
   * Detect circular dependencies in the task graph
   */
  function detectCycles(): string[][] {
    const cycles: string[][] = [];
    const allTasks = taskStore.list();
    const visited = new Set<string>();
    const recursionStack = new Set<string>();
    const path: string[] = [];

    function dfs(taskId: string): boolean {
      visited.add(taskId);
      recursionStack.add(taskId);
      path.push(taskId);

      const deps = taskStore.getDependencies(taskId);
      for (const depId of deps) {
        if (!visited.has(depId)) {
          if (dfs(depId)) {
            return true;
          }
        } else if (recursionStack.has(depId)) {
          // Found cycle - extract it
          const cycleStart = path.indexOf(depId);
          if (cycleStart !== -1) {
            cycles.push([...path.slice(cycleStart), depId]);
          } else {
            cycles.push([...path, depId]);
          }
          return true;
        }
      }

      path.pop();
      recursionStack.delete(taskId);
      return false;
    }

    for (const task of allTasks) {
      if (!visited.has(task.id)) {
        dfs(task.id);
      }
    }

    return cycles;
  }

  /**
   * Get the critical path (longest path by estimated duration)
   */
  function getCriticalPath(rootTaskId: string): CriticalPathResult {
    const graph = buildGraph(rootTaskId);
    const durations = new Map<string, number>();
    const predecessors = new Map<string, string | null>();

    // Topological sort
    const sorted = topologicalSortInternal(graph);

    // Initialize
    for (const [taskId] of graph.nodes) {
      durations.set(taskId, 0);
      predecessors.set(taskId, null);
    }

    // Calculate longest path
    for (const taskId of sorted) {
      const node = graph.nodes.get(taskId);
      if (!node) continue;

      const currentDuration = durations.get(taskId)!;
      const taskDuration = node.task.estimatedDuration || 0;

      for (const depId of node.dependencies) {
        const depNode = graph.nodes.get(depId);
        if (!depNode) continue;

        const depDuration = depNode.task.estimatedDuration || 0;
        const newDuration = currentDuration + depDuration;

        if (newDuration > durations.get(depId)!) {
          durations.set(depId, newDuration);
          predecessors.set(depId, taskId);
        }
      }
    }

    // Find end node (task with no dependents in our subgraph)
    let endTaskId = rootTaskId;
    let maxDuration = 0;

    for (const [taskId, duration] of durations.entries()) {
      if (duration > maxDuration) {
        maxDuration = duration;
        endTaskId = taskId;
      }
    }

    // Reconstruct path
    const path: string[] = [];
    let current: string | null = endTaskId;

    while (current) {
      path.unshift(current);
      current = predecessors.get(current) || null;
    }

    return {
      path,
      totalEstimatedDuration: maxDuration + (graph.nodes.get(endTaskId)?.task.estimatedDuration || 0)
    };
  }

  /**
   * Topological sort of all tasks
   */
  function topologicalSort(): string[] {
    const graph = buildGraph();
    return topologicalSortInternal(graph);
  }

  function topologicalSortInternal(graph: TaskGraph): string[] {
    const result: string[] = [];
    const visited = new Set<string>();
    const temp = new Set<string>();

    function visit(taskId: string): void {
      if (temp.has(taskId)) return; // Cycle, skip
      if (visited.has(taskId)) return;

      temp.add(taskId);

      const node = graph.nodes.get(taskId);
      if (node) {
        for (const depId of node.dependencies) {
          visit(depId);
        }
      }

      temp.delete(taskId);
      visited.add(taskId);
      result.push(taskId);
    }

    for (const [taskId] of graph.nodes) {
      if (!visited.has(taskId)) {
        visit(taskId);
      }
    }

    return result;
  }

  /**
   * Get tasks that can be executed in parallel (same depth, no dependencies between them)
   */
  function getParallelGroups(rootTaskId: string): string[][] {
    const graph = buildGraph(rootTaskId);
    const groups: string[][] = [];
    const assigned = new Set<string>();

    function getDepth(taskId: string, visited = new Set<string>()): number {
      if (visited.has(taskId)) return 0;
      visited.add(taskId);

      const node = graph.nodes.get(taskId);
      if (!node) return 0;

      if (node.dependencies.length === 0) {
        return 0;
      }

      let maxDepDepth = 0;
      for (const depId of node.dependencies) {
        maxDepDepth = Math.max(maxDepDepth, getDepth(depId, visited) + 1);
      }

      return maxDepDepth;
    }

    // Assign depths
    const depths = new Map<string, number>();
    for (const [taskId] of graph.nodes) {
      depths.set(taskId, getDepth(taskId));
    }

    // Group by depth
    const depthGroups = new Map<number, string[]>();
    for (const [taskId, depth] of depths.entries()) {
      if (!depthGroups.has(depth)) {
        depthGroups.set(depth, []);
      }
      depthGroups.get(depth)!.push(taskId);
    }

    // Sort by depth and add to groups
    const sortedDepths = Array.from(depthGroups.keys()).sort((a, b) => a - b);
    for (const depth of sortedDepths) {
      const tasks = depthGroups.get(depth)!;
      const executableInGroup = tasks.filter(taskId => {
        const node = graph.nodes.get(taskId);
        if (!node) return false;

        // Check if all dependencies are in previous groups (completed)
        for (const depId of node.dependencies) {
          const depDepth = depths.get(depId);
          if (depDepth !== undefined && depDepth >= depth) {
            return false;
          }
        }
        return true;
      });

      if (executableInGroup.length > 0) {
        groups.push(executableInGroup);
      }
    }

    return groups;
  }

  return {
    buildGraph,
    getExecutableTasks,
    getTaskPath,
    detectCycles,
    getCriticalPath,
    topologicalSort,
    getParallelGroups
  };
}
