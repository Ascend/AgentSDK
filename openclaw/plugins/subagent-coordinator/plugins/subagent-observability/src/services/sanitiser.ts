import type { SubagentCoordinatorEventName } from "@subagent-coordinator/types";

export interface SanitiserConfig {
  enabled: boolean;
  replacement: string;
  patterns: SanitisationPattern[];
  customRules: CustomSanitisationRule[];
}

export interface SanitisationPattern {
  name: string;
  pattern: string;
  replacement?: string;
}

export interface CustomSanitisationRule {
  name: string;
  matcher: (value: string) => boolean;
  replacement: string;
}

export interface SanitiseResult {
  sanitised: boolean;
  value: string;
  redactedCount: number;
  redactedTypes: string[];
}

export interface SanitiserState {
  config: SanitiserConfig;
}

const DEFAULT_PATTERNS: SanitisationPattern[] = [
  {
    name: "openai_api_key",
    pattern: "sk-[A-Za-z0-9]{20,}[A-Za-z0-9_-]*",
  },
  {
    name: "anthropic_api_key",
    pattern: "sk-ant-[A-Za-z0-9_-]{20,}",
  },
  {
    name: "aws_access_key",
    pattern: "(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{20,40}(?![A-Za-z0-9/+=])",
  },
  {
    name: "aws_secret_key",
    pattern: "(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])",
  },
  {
    name: "bearer_token",
    pattern: "bearer\\s+[A-Za-z0-9_-]{20,}",
  },
  {
    name: "github_token",
    pattern: "gh[pousr]_[A-Za-z0-9_]{36,}",
  },
  {
    name: "generic_api_key",
    pattern: "(?<![A-Za-z0-9])[a-zA-Z0-9_-]{30,50}(?![A-Za-z0-9/+=])",
  },
  {
    name: "jwt_token",
    pattern: "eyJ[A-Za-z0-9_-]*\\.eyJ[A-Za-z0-9_-]*\\.[A-Za-z0-9_-]*",
  },
  {
    name: "password_in_url",
    pattern: "://[^:]+:[^@]+@",
  },
  {
    name: "private_key",
    pattern: "-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
  },
];

const DEFAULT_CONFIG: SanitiserConfig = {
  enabled: true,
  replacement: "[REDACTED]",
  patterns: DEFAULT_PATTERNS,
  customRules: [],
};

export interface SanitiserService {
  sanitise(value: string): SanitiseResult;
  sanitiseObject(obj: unknown): SanitiseResult;
  containsSensitive(value: string): boolean;
  detectTypes(value: string): string[];
  updatePatterns(patterns: SanitisationPattern[]): void;
  addCustomRule(rule: CustomSanitisationRule): void;
  setEnabled(enabled: boolean): void;
}

export function createSanitiser(
  state: SanitiserState,
  config: Partial<SanitiserConfig> = {}
): SanitiserService {
  state.config = {
    ...DEFAULT_CONFIG,
    ...config,
    patterns: [...DEFAULT_PATTERNS, ...(config.patterns || [])],
    customRules: [...(config.customRules || [])],
  };

  const compiledPatterns = state.config.patterns.map((p) => ({
    ...p,
    regex: new RegExp(p.pattern, "gi"),
  }));

  const performSanitise = (value: string): SanitiseResult => {
    if (!state.config.enabled) {
      return {
        sanitised: false,
        value,
        redactedCount: 0,
        redactedTypes: [],
      };
    }

    let sanitisedValue = value;
    const redactedTypes: Set<string> = new Set();
    let redactedCount = 0;

    for (const { name, regex, replacement } of compiledPatterns) {
      const replacementStr = replacement || state.config.replacement;
      const before = sanitisedValue;
      sanitisedValue = sanitisedValue.replace(regex, replacementStr);

      if (sanitisedValue !== before) {
        const matches = before.match(regex);
        if (matches) {
          redactedCount += matches.length;
          redactedTypes.add(name);
        }
      }
      regex.lastIndex = 0;
    }

    for (const rule of state.config.customRules) {
      if (rule.matcher(sanitisedValue)) {
        sanitisedValue = sanitisedValue.replace(
          new RegExp(escapeRegExp(sanitisedValue), "g"),
          rule.replacement
        );
        redactedTypes.add(rule.name);
        redactedCount++;
      }
    }

    return {
      sanitised: redactedCount > 0,
      value: sanitisedValue,
      redactedCount,
      redactedTypes: Array.from(redactedTypes),
    };
  };

  return {
    sanitise(value) {
      return performSanitise(value);
    },

    sanitiseObject(obj) {
      if (!state.config.enabled) {
        return {
          sanitised: false,
          value: JSON.stringify(obj),
          redactedCount: 0,
          redactedTypes: [],
        };
      }

      const stringified = JSON.stringify(obj);
      return performSanitise(stringified);
    },

    containsSensitive(value) {
      if (!state.config.enabled) return false;

      for (const { regex } of compiledPatterns) {
        if (regex.test(value)) {
          return true;
        }
        regex.lastIndex = 0;
      }

      return false;
    },

    detectTypes(value) {
      if (!state.config.enabled) return [];

      const types: string[] = [];

      for (const { name, regex } of compiledPatterns) {
        if (regex.test(value)) {
          types.push(name);
        }
        regex.lastIndex = 0;
      }

      return types;
    },

    updatePatterns(patterns) {
      state.config.patterns = [...DEFAULT_PATTERNS, ...patterns];
      compiledPatterns.length = 0;
      for (const p of state.config.patterns) {
        compiledPatterns.push({
          ...p,
          regex: new RegExp(p.pattern, "gi"),
        });
      }
    },

    addCustomRule(rule) {
      state.config.customRules.push(rule);
    },

    setEnabled(enabled) {
      state.config.enabled = enabled;
    },
  };
}

function escapeRegExp(string: string): string {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function createSanitiserState(): SanitiserState {
  return {
    config: DEFAULT_CONFIG,
  };
}
