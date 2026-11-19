# Kris — Research Notes

一个面向研究笔记与技术写作的个人静态博客，使用 Astro 构建并通过 GitHub Pages 发布。

## 为什么选择 Astro

- 默认生成静态 HTML，适合 GitHub Pages；
- 原生支持 Markdown，并用 Content Collections 校验文章元数据；
- 设计自由度高，同时默认只发送必要的 JavaScript；
- 后续需要 MDX 或少量交互时，不必迁移框架。

如果只需要套用传统主题，Jekyll 会更直接；如果已经熟悉 Go Template，Hugo 也是很好的选择。这个仓库更强调自定义阅读体验，因此使用 Astro。

## 本地运行

需要 Node.js 22 或更高版本。

```bash
npm install
npm run dev
```

构建与预览：

```bash
npm run build
npm run preview
```

## 写一篇文章

在 `src/content/posts` 新建 Markdown 文件：

```md
---
title: 文章标题
description: 一句话摘要
publishDate: 2026-11-19
tags: [研究方法, 实验]
draft: false
featured: false
---

从这里开始写正文。
```

现有三篇内容都设置了 `isExample: true`，用于展示页面。正式发布前可删除它们，或改写后移除该字段。

## 常用修改位置

| 需要修改的内容 | 文件 |
| --- | --- |
| 站点名称、描述、GitHub 地址 | `src/config.ts` |
| 首页介绍和版块文案 | `src/pages/index.astro` |
| 个人介绍与关注方向 | `src/pages/about.astro` |
| 颜色、字体与通用样式 | `src/styles/global.css` |
| 文章字段规则 | `src/content.config.ts` |

## 发布到 GitHub Pages

仓库已经包含 `.github/workflows/deploy.yml`。首次发布前：

1. 打开 GitHub 仓库的 **Settings → Pages**；
2. 在 **Build and deployment → Source** 中选择 **GitHub Actions**；
3. 将代码推送到 `main` 分支；
4. 等待 `Deploy to GitHub Pages` 工作流完成。

用户站点的地址是 <https://krisyang7.github.io>。这个仓库不需要设置 Astro 的 `base` 路径。

如果以后使用自定义域名，请在 `public/CNAME` 写入域名，并同步修改 `astro.config.mjs` 和 `src/config.ts` 中的网址。
