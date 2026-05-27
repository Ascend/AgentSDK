import type { SanitiserService } from "../services/sanitiser";

export function createDetectSensitiveTool(sanitiser: SanitiserService) {
  return async (input: { value: string }): Promise<{
    containsSensitive: boolean;
    detectedTypes: string[];
  }> => {
    const containsSensitive = sanitiser.containsSensitive(input.value);
    const detectedTypes = sanitiser.detectTypes(input.value);

    return {
      containsSensitive,
      detectedTypes,
    };
  };
}
