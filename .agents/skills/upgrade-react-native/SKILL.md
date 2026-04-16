---
name: upgrade-react-native
description: Upgrade React Native version in example/ directory using rn-diff-purge methodology. Use when the user wants to upgrade react-native, migrate to a new RN version, or update React Native dependencies in the example app. Handles diff retrieval, systematic file changes, dependency updates, and conflict resolution.
---

# Upgrade React Native

## CRITICAL: File Editing Rules

- **ALWAYS use Edit/Write tools** to make changes - never just report what should be changed
- **VERIFY changes were saved** by reading the file after editing
- **Actually apply all changes** - do not just list them in a summary

## Workflow

### 1. Detect Current Version

Read `example/package.json` to identify the current react-native version.

### 2. Fetch Upgrade Diff

```bash
curl -s "https://raw.githubusercontent.com/react-native-community/rn-diff-purge/diffs/diffs/{currentVersion}..{toVersion}.diff" | awk '/^GIT binary patch$/,/^diff --git/ {if (/^diff --git/) print; next} 1'
```

For binary files (e.g., gradle-wrapper.jar):
```
https://raw.githubusercontent.com/react-native-community/rn-diff-purge/release/{version}/RnDiffApp/android/gradle/wrapper/gradle-wrapper.jar
```

If diff returns 404: version combination may not exist in rn-diff-purge; suggest alternative paths or manual migration.

### 3. Analyze Diff

Identify file additions/deletions/modifications, dependency changes, config updates, native code changes (iOS + Android), and breaking changes.

### 4. Apply Changes Systematically

Order: package.json deps -> config files -> code changes -> iOS native -> Android native. Preserve custom modifications in example/.

### 5. Update Related Packages

- `react` - Check: `npm view react-native@{version} peerDependencies.react`
- `react-test-renderer` - Must match React version
- `@react-native-community/cli` - Check compatibility with RN version

### 6. Install Dependencies

```bash
cd example && npm install
```

For iOS:
```bash
cd example/ios && pod install
```

### 7. Validate

Rebuild native apps from root directory:
```bash
npm run build:ios
npm run build:android
```

## Guidelines

- **Incremental upgrades**: For multiple major versions, upgrade incrementally (e.g., 0.72 -> 0.73 -> 0.74).
- **Conflict resolution**: Preserve custom functionality, explain conflicts, recommend safest resolution.
- **Never delete custom code** without explicit confirmation.
- **Only work within example/** unless explicitly instructed otherwise.
- **Document all changes**: Summarize modified/added/deleted files, breaking changes, and manual steps needed.
