import type { SanitiserService } from "../services/sanitiser";

export function createSanitiseDataTool(sanitiser: SanitiserService) {
  return async (input: {
    value: string;
    isObject?: boolean;
  }): Promise<{
    sanitised: boolean;
    value: string;
    redactedCount: number;
    redactedTypes: string[];
  }> => {
    if (input.isObject) {
      return sanitiser.sanitiseObject(JSON.parse(input.value));
    }
    return sanitiser.sanitise(input.value);
  };
}
