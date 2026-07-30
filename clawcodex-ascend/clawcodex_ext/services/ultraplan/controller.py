"""Controller that wires planner, store, executor, CCR, and audit."""

from __future__ import annotations

from pathlib import Path

from .audit import AuditLogger
from .ccr_session import CCRClient
from .exceptions import CCRUnavailableError, PlanNotFoundError
from .executor import PlanExecutor, Progress
from .llm_planner import LLMPlanner, PlannerContext, PlannerResult
from .models import Plan, PlanStatus
from .store import PlanStore
from .verifier import AcceptanceVerifier


class UltraplanController:
    def __init__(
        self,
        *,
        planner: LLMPlanner | None,
        store: PlanStore,
        audit: AuditLogger | None = None,
        ccr: CCRClient | None = None,
    ) -> None:
        self.planner = planner
        self.store = store
        self.audit = audit
        self.ccr = ccr

    async def create_plan(self, ctx: PlannerContext) -> PlannerResult:
        if self.planner is None:
            raise RuntimeError("ultraplan planner is not configured")
        result = await self.planner.generate_plan(ctx)
        self.store.save(result.plan)
        if self.audit is not None:
            self.audit.append(
                result.plan.id,
                "plan.created",
                {
                    "title": result.plan.title,
                    "provider": result.provider,
                    "model": result.model,
                    "retry_count": result.retry_count,
                },
            )
        return result

    async def run_plan(self, plan_id: str, *, remote: bool = False, cwd: str | None = None) -> Progress:
        plan = self.store.load(plan_id)
        if remote:
            if self.ccr is None:
                raise CCRUnavailableError("CCR remote execution is not configured")
            session = await self.ccr.start_session(plan, cwd=cwd or str(Path.cwd()))
            if self.audit is not None:
                self.audit.append(plan.id, "ccr.session.started", {"session_id": session.id})
            return PlanExecutor(plan).progress()
        hooks = [self.audit.record_transition] if self.audit is not None else None
        executor = PlanExecutor(plan, transition_hooks=hooks)
        verifier = AcceptanceVerifier(executor.plan)
        made_progress = True
        while made_progress and executor.plan.status != PlanStatus.PAUSED:
            made_progress = False
            step = executor.next_step()
            if step is None:
                break
            executor.mark_in_progress(step.id)
            criteria_results = verifier.verify_step(step.id) if step.criteria else {}
            required_failures = {
                criterion_id: result
                for criterion_id, result in criteria_results.items()
                if not result.passed and any(c.id == criterion_id and c.required for c in step.criteria)
            }
            if required_failures:
                details = "; ".join(
                    f"{criterion_id}: {result.details}" for criterion_id, result in required_failures.items()
                )
                executor.mark_failed(step.id, details or "required acceptance criteria failed")
            else:
                executor.mark_completed(
                    step.id,
                    result={
                        "criteria": {
                            criterion_id: result.to_dict() for criterion_id, result in criteria_results.items()
                        }
                    },
                    note="advanced by /ultraplan run",
                )
            self.store.save(executor.plan)
            if self.audit is not None and criteria_results:
                self.audit.append(
                    plan.id,
                    "step.criteria",
                    {
                        "step_id": step.id,
                        "results": {
                            criterion_id: result.to_dict() for criterion_id, result in criteria_results.items()
                        },
                    },
                )
            made_progress = True
            if required_failures:
                break
        self.store.save(executor.plan)
        return executor.progress()

    async def pause_plan(self, plan_id: str) -> Plan:
        plan = self.store.load(plan_id)
        plan.status = PlanStatus.PAUSED
        self.store.save(plan)
        if self.audit is not None:
            self.audit.append(plan.id, "plan.paused", {})
        return plan

    async def resume_plan(self, plan_id: str) -> Plan:
        plan = self.store.load(plan_id)
        plan.status = PlanStatus.ACTIVE
        self.store.save(plan)
        if self.audit is not None:
            self.audit.append(plan.id, "plan.resumed", {})
        return plan

    async def list_plans(self, *, status: PlanStatus | None = None) -> list[Plan]:
        plans: list[Plan] = []
        for plan_id, plan_status, _title in self.store.list_plans():
            if status is not None and plan_status != status:
                continue
            try:
                plans.append(self.store.load(plan_id))
            except PlanNotFoundError:
                continue
        return plans

    async def show_plan(self, plan_id: str) -> Plan:
        return self.store.load(plan_id)

    async def remove_plan(self, plan_id: str) -> None:
        plan = self.store.load(plan_id)
        if plan.status not in {PlanStatus.DRAFT, PlanStatus.ABANDONED, PlanStatus.FAILED}:
            raise ValueError("only draft, abandoned, or failed plans can be removed")
        self.store.delete(plan_id)
        if self.audit is not None:
            self.audit.append(plan.id, "plan.removed", {})
