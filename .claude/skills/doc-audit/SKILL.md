```markdown
---
name: doc-audit
description: 审计项目 CLAUDE.md 与实际代码结构的一致性。对比项目结构声明、依赖拓扑、模块状态标记，标记漂移。适用于重构后、部署前确认文档是否最新。
allowed-tools: Read, Grep, Glob, Agent
---

## 执行步骤

spawn **2 个 Explore Agent** 并行审计，然后汇总：

### Agent 1 — 项目结构一致性
```

对比 CLAUDE.md 中声明的项目结构树与实际代码目录。

流程：

1. 读取 CLAUDE.md，提取"项目结构"章节中的目录树（所有列出的目录和文件）
2. Glob 扫描项目实际存在的目录和 Python 文件
3. 逐项对比，标记：
   - CLAUDE.md 声明但实际不存在的目录/文件
   - 实际存在但 CLAUDE.md 未提及的目录/文件
4. 检查 ✅ 状态标记：对比 CLAUDE.md 中的状态描述与模块代码的实际完成度

输出格式：

### 结构一致性

#### 声明但不存在（[N] 项）

- CLAUDE.md:L行号 — `目录名/` — 未找到

#### 存在但未声明（[N] 项）

- `实际路径` — 未在 CLAUDE.md 中出现

#### 状态标记检查

- ✅ 正确: [N] 项
- ⚠️ 存疑: CLAUDE.md:L行号 — `模块名` — 原因

边界情况：

- 无 CLAUDE.md: 报告并退出
- 空项目: 报告并退出



```
### Agent 2 — 依赖拓扑一致性
```

对比 CLAUDE.md 中声明的依赖拓扑与实际代码 import 关系。

流程：

1. 读取 CLAUDE.md，提取"依赖拓扑"章节的依赖图
2. Grep 扫描全项目 Python 文件的 import 语句
3. 构建实际依赖图
4. 逐边对比，标记：
   - 拓扑声明了但代码中不存在的依赖边
   - 代码中存在但拓扑未声明的依赖边

输出格式：

### 依赖拓扑一致性

#### 声明但未引用（[N] 项）

- 拓扑: A→B — 代码中 A 未 import B

#### 引用但未声明（[N] 项）

- 代码: `模块A/file.py:L行号` — `from 模块B import ...`
- 拓扑中无此依赖边

#### 验证通过的依赖边（[N] 项）

- A→B ✅
- B→C ✅

边界情况：

- CLAUDE.md 无依赖拓扑章节: 报告并跳过
- 无 Python 文件: 报告并退出



```
### 汇总输出

两个 Agent 完成后，汇总为：

## 文档一致性审计报告

### 概要
- CLAUDE.md 声明模块: [N] | 实际: [N] | 匹配: [N]
- 依赖边声明: [N] | 实际 import: [N] | 匹配: [N]

### 不一致项
[标记为 MISSING_IN_DOC / MISSING_IN_CODE / WRONG_DEPENDENCY / STALE_STATUS]

### 建议
- [优先级 1]
- [优先级 2]

> 只报告不一致，不要自动修改任何文件。最终由人判断。
```