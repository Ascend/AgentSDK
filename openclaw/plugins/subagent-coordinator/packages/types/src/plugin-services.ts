/*!
 * This file is part of the AgentSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 *
 * AgentSDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

import type { Task, ComplexityScore, OperatorLevel, Subtask, QualityGateResult, CheckpointData, DecompositionStrategy } from "./types";

export interface PluginServices {
  taskComplexityScorer?: {
    score(task: Task): Promise<ComplexityScore>;
  };

  operatorClassifier?: {
    classify(complexity: ComplexityScore): Promise<OperatorLevel>;
  };

  taskDecomposer?: {
    decompose(task: Task, strategy: DecompositionStrategy): Promise<Subtask[]>;
    persist(task: Task): Promise<string>;
  };

  checkpointManager?: {
    save(checkpoint: CheckpointData): Promise<string>;
    restore(checkpointId: string): Promise<CheckpointData | null>;
    list(taskId: string): Promise<string[]>;
  };

  qualityGate?: {
    check(task: Task, preExecution: boolean): Promise<QualityGateResult>;
  };
}
