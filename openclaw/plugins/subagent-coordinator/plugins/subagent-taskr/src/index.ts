import { Type } from "@sinclair/typebox";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

import type { Task, Subtask, DecompositionStrategy, ComplexityScore, OperatorLevel } from "@subagent-coordinator/types";

export { SUBAGENT_COORDINATOR_EVENTS } from "@subagent-coordinator/types";
export type {
  Task,
  Subtask,
  DecompositionStrategy,
  CheckpointData,
  BeforeDelegationEvent,
  AfterExecutionEvent,
  TaskAnalyzedEvent,
  DecompositionRequestedEvent,
  CheckpointSaveEvent,
  CheckpointRestoreEvent,
} from "@subagent-coordinator/types";

import {
  type TaskStoreService,
} from "./services/task_store";

import {
  createMemoryTaskStore,
  type MemoryPersistenceService,
} from "./services/memory_task_store";

import {
  createTaskGraph,
  type TaskGraphState,
  type TaskGraphService,
} from "./services/task_graph";

import {
  HeuristicComplexityScorer,
  type ComplexityScorer,
} from "./services/complexity_scorer";

interface TaskrPluginState {
  taskGraph: TaskGraphState;
  checkpoints: Map<string, import("@subagent-coordinator/types").CheckpointData>;
}

let _state: TaskrPluginState | null = null;
let _taskStore: TaskStoreService & { getMemoryStore?: () => MemoryPersistenceService } | null = null;
let _taskGraph: TaskGraphService | null = null;
let _scorer: ComplexityScorer | null = null;

function getState(): TaskrPluginState {
  if (!_state) {
    _state = {
      taskGraph: { nodes: new Map() },
      checkpoints: new Map(),
    };
  }
  return _state;
}

function getTaskStore(): TaskStoreService & { getMemoryStore?: () => MemoryPersistenceService } {
  if (!_taskStore) {
    _taskStore = createMemoryTaskStore({ warnIndexDelay: false });
  }
  return _taskStore;
}

function getTaskGraph(): TaskGraphService {
  if (!_taskGraph) {
    _taskGraph = createTaskGraph(getTaskStore());
  }
  return _taskGraph;
}

function getScorer(): ComplexityScorer {
  if (!_scorer) {
    _scorer = new HeuristicComplexityScorer();
  }
  return _scorer;
}

import { createCreateTaskTool } from "./tools/create_task";
import { createGetTaskTool } from "./tools/get_task";
import { createUpdateTaskTool } from "./tools/update_task";
import { createListTasksTool } from "./tools/list_tasks";
import { createDecomposeTaskTool } from "./tools/decompose_task";
import { createAddNoteTool } from "./tools/add_note";
import { createScoreComplexityTool } from "./tools/score_complexity";
import { createClassifyOperatorTool } from "./tools/classify_operator";

export default definePluginEntry({
  id: "@subagent-coordinator/taskr",
  name: "Subagent Coordinator Taskr",
  description: "Hierarchical task planning, persistent state management, and cross-agent continuity",

  register(api) {
    const taskStore = getTaskStore();
    const taskGraph = getTaskGraph();
    const scorer = getScorer();

    api.registerTool({
      name: "create_task",
      label: "create_task",
      description: "Create a new task with optional parent, priority, and metadata",
      parameters: Type.Object({
        description: Type.String({ description: "Task description" }),
        parentId: Type.Optional(Type.String({ description: "Parent task ID for hierarchy" })),
        priority: Type.Optional(Type.Union([
          Type.Literal("low"),
          Type.Literal("normal"),
          Type.Literal("high"),
          Type.Literal("urgent"),
        ], { description: "Task priority" })),
        estimatedDuration: Type.Optional(Type.Number({ description: "Estimated duration in seconds" })),
        metadata: Type.Optional(Type.Record(Type.String(), Type.Unknown(), { description: "Additional metadata" })),
        steps: Type.Optional(Type.Number({ description: "Number of steps" })),
        files: Type.Optional(Type.Array(Type.String(), { description: "Files involved" })),
      }),
      async execute(_id, params) {
        const handler = createCreateTaskTool(taskStore);
        const result = await handler(params as any);
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "get_task",
      label: "get_task",
      description: "Retrieve a task by ID with notes, dependencies, and children",
      parameters: Type.Object({
        taskId: Type.String({ description: "Task ID to retrieve" }),
        includeNotes: Type.Optional(Type.Boolean({ description: "Include task notes" })),
        includeDependencies: Type.Optional(Type.Boolean({ description: "Include dependency IDs" })),
        includeChildren: Type.Optional(Type.Boolean({ description: "Include child subtasks" })),
        includePath: Type.Optional(Type.Boolean({ description: "Include task path from root" })),
      }),
      async execute(_id, params) {
        const handler = createGetTaskTool(taskStore, taskGraph);
        const result = await handler(params as any);
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "update_task",
      label: "update_task",
      description: "Update task properties including status transitions (open → wip → done/skipped)",
      parameters: Type.Object({
        taskId: Type.String({ description: "Task ID to update" }),
        description: Type.Optional(Type.String({ description: "New task description" })),
        status: Type.Optional(Type.Union([
          Type.Literal("open"),
          Type.Literal("wip"),
          Type.Literal("done"),
          Type.Literal("skipped"),
        ], { description: "New task status" })),
        priority: Type.Optional(Type.Union([
          Type.Literal("low"),
          Type.Literal("normal"),
          Type.Literal("high"),
          Type.Literal("urgent"),
        ])),
        estimatedDuration: Type.Optional(Type.Number()),
        steps: Type.Optional(Type.Number()),
        files: Type.Optional(Type.Array(Type.String())),
        metadata: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
      }),
      async execute(_id, params) {
        const handler = createUpdateTaskTool(taskStore);
        const result = await handler(params as any);
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "list_tasks",
      label: "list_tasks",
      description: "List tasks with filtering, sorting, and pagination",
      parameters: Type.Object({
        status: Type.Optional(Type.Union([
          Type.Literal("open"),
          Type.Literal("wip"),
          Type.Literal("done"),
          Type.Literal("skipped"),
        ])),
        parentId: Type.Optional(Type.Union([Type.String(), Type.Null()], { description: "Filter by parent (null = root tasks)" })),
        priority: Type.Optional(Type.Union([
          Type.Literal("low"),
          Type.Literal("normal"),
          Type.Literal("high"),
          Type.Literal("urgent"),
        ])),
        tags: Type.Optional(Type.Array(Type.String())),
        createdAfter: Type.Optional(Type.Number()),
        createdBefore: Type.Optional(Type.Number()),
        sortBy: Type.Optional(Type.Union([
          Type.Literal("createdAt"),
          Type.Literal("updatedAt"),
          Type.Literal("priority"),
          Type.Literal("estimatedDuration"),
        ])),
        sortOrder: Type.Optional(Type.Union([Type.Literal("asc"), Type.Literal("desc")])),
        limit: Type.Optional(Type.Number({ default: 50 })),
        offset: Type.Optional(Type.Number({ default: 0 })),
      }),
      async execute(_id, params) {
        const handler = createListTasksTool(taskStore, taskGraph);
        const result = await handler(params as any);
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "decompose_task",
      label: "decompose_task",
      description: "Decompose a task into subtasks using by_file, by_step, or by_domain strategy",
      parameters: Type.Object({
        taskId: Type.String({ description: "Task ID to decompose" }),
        strategy: Type.Union([
          Type.Literal("by_file"),
          Type.Literal("by_step"),
          Type.Literal("by_domain"),
        ], { description: "Decomposition strategy" }),
        maxSubtasks: Type.Optional(Type.Number({ default: 10 })),
      }),
      async execute(_id, params) {
        const handler = createDecomposeTaskTool(taskStore, taskGraph);
        const result = await handler(params as any);
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "add_note",
      label: "add_note",
      description: "Add a note to a task for context, findings, progress, or errors",
      parameters: Type.Object({
        taskId: Type.String({ description: "Task ID" }),
        content: Type.String({ description: "Note content" }),
        type: Type.Union([
          Type.Literal("context"),
          Type.Literal("finding"),
          Type.Literal("progress"),
          Type.Literal("file_list"),
          Type.Literal("error"),
        ], { description: "Note type" }),
        author: Type.Optional(Type.String()),
      }),
      async execute(_id, params) {
        const handler = createAddNoteTool(taskStore);
        const result = await handler(params as any);
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "score_complexity",
      label: "score_complexity",
      description: "Score task complexity based on multiple factors (steps, files, dependency, determinism, keywords)",
      parameters: Type.Object({
        task: Type.Object({
          id: Type.String(),
          description: Type.String(),
          steps: Type.Optional(Type.Number()),
          files: Type.Optional(Type.Array(Type.String())),
          estimatedDuration: Type.Optional(Type.Number()),
          priority: Type.Optional(Type.Union([
            Type.Literal("low"),
            Type.Literal("normal"),
            Type.Literal("high"),
            Type.Literal("urgent"),
          ])),
        }),
      }),
      async execute(_id, params) {
        const handler = createScoreComplexityTool(scorer);
        const result = await handler(params as any);
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "classify_operator",
      label: "classify_operator",
      description: "Classify operator level (L1-L5) based on complexity score",
      parameters: Type.Object({
        complexity: Type.Object({
          total: Type.Number(),
          breakdown: Type.Object({
            steps: Type.Number(),
            files: Type.Number(),
            dependency: Type.Number(),
            determinism: Type.Number(),
          }),
          keywords: Type.Array(Type.String()),
        }),
      }),
      async execute(_id, params) {
        const handler = createClassifyOperatorTool(scorer);
        const result = await handler(params as any);
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });
  },
});
