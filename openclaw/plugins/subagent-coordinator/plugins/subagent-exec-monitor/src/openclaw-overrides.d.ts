// Suppress type errors from openclaw internal source files that are
// transitively loaded via openclaw/plugin-sdk re-exports.
// These errors are in openclaw's internal modules, not in plugin code.
declare module "openclaw/src/*" {
  const anyExport: any;
  export = anyExport;
}

declare module "openclaw/src/agents/*" {
  const anyExport: any;
  export = anyExport;
}

declare module "openclaw/src/tasks/*" {
  const anyExport: any;
  export = anyExport;
}

declare module "openclaw/src/plugins/*" {
  const anyExport: any;
  export = anyExport;
}

declare module "openclaw/src/process/*" {
  const anyExport: any;
  export = anyExport;
}

declare module "openclaw/src/shared/*" {
  const anyExport: any;
  export = anyExport;
}

declare module "openclaw/src/proxy-capture/*" {
  const anyExport: any;
  export = anyExport;
}

declare module "openclaw/src/acp/*" {
  const anyExport: any;
  export = anyExport;
}
