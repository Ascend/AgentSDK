import type { CostTrackerService, BudgetStatus } from "../services/cost_tracker";

export function createGetBudgetStatusTool(costTracker: CostTrackerService) {
  return async (input?: {}): Promise<{
    budget: BudgetStatus;
    alert: { shouldAlert: boolean; message?: string };
  }> => {
    const budget = costTracker.getBudgetStatus();
    const alert = costTracker.checkBudgetAlert();

    return {
      budget,
      alert,
    };
  };
}
