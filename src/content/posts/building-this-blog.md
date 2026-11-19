---
title: 这个博客是如何搭起来的
description: 从框架选择、笔记结构到自动部署，记录一个个人学习档案的最小实现。
publishDate: 2026-11-19
tags: [建站, Astro, 工作流]
noteType: learning
readingStatus: finished
featured: false
isExample: true
---

> 这是一篇用于展示笔记排版与内容结构的示例内容。正式使用前，可以直接替换或删除它。

个人学习档案最重要的不是功能数量，而是让“记下一次阅读”这件事足够简单。这个网站因此只保留了几个核心目标：内容使用 Markdown 管理、笔记可以搜索和筛选、页面默认静态生成、代码推送后自动部署。

## 为什么选择 Astro

可选方案很多：Jekyll 与 GitHub Pages 结合紧密，Hugo 的构建速度很快，而 Astro 在内容管理和前端设计之间提供了更自然的平衡。

这里最终选择 Astro，主要因为：

- Markdown 和 MDX 都是一等内容源；
- 构建结果是纯静态文件，适合 GitHub Pages；
- 默认不会把不必要的 JavaScript 发到浏览器；
- 页面结构可以继续使用熟悉的 HTML 与 CSS。

这不是一份通用排名。如果只需要传统主题和最少定制，Jekyll 依然直接；如果已有 Hugo 使用经验，也没有必要为了框架而迁移。

## 内容如何组织

每篇笔记是 `src/content/posts` 下的一个文件。文件开头的 frontmatter 描述标题、摘要、日期、类型与标签：

```yaml
---
title: 一篇新笔记
description: 用一句话说明以后为什么值得回看。
publishDate: 2026-11-19
tags: [论文阅读, Transformer]
noteType: learning
draft: false
---
```

字段由 Content Collection 统一校验。写错日期或漏掉必填字段时，构建阶段就会给出提示，而不是等到部署后才发现页面异常。

## 部署路径

当代码被推送到 `main` 分支时，GitHub Actions 会执行构建并把 `dist` 目录发布到 Pages。整个过程可以概括为：

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| 写作 | Markdown | 结构化内容 |
| 构建 | Astro 项目 | 静态 HTML/CSS |
| 发布 | `dist` 目录 | GitHub Pages 网站 |

对于 `username.github.io` 形式的用户站点，页面直接位于域名根路径，因此无需再设置项目级 `base`。

## 保持最小

首版只加入了浏览器端的关键词搜索与类型筛选，没有评论、访问统计或复杂知识图谱。这些功能并非永远不需要，只是应该在真实需求出现后再加入。笔记数量不多时，搜索、标签和时间归档已经足够。

网站的结构也可以随着内容一起成长，而不是在第一天就为想象中的规模付出维护成本。
