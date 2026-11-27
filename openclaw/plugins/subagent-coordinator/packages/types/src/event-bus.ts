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

import type { SubagentCoordinatorEventName } from "./events";
import type { GetEventPayload } from "./payload-map";

export interface EventBus {
  emit<T extends SubagentCoordinatorEventName>(
    event: T,
    payload: GetEventPayload<T>
  ): Promise<void>;

  on<T extends SubagentCoordinatorEventName>(
    event: T,
    handler: (payload: GetEventPayload<T>) => Promise<void>
  ): void;

  off<T extends SubagentCoordinatorEventName>(
    event: T,
    handler: (payload: GetEventPayload<T>) => Promise<void>
  ): void;
}
