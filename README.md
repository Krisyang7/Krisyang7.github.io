# Kris — Learning Notes

一个用于长期保存论文阅读、学习记录和阶段性思考的个人知识档案，使用 Astro 构建并通过 GitHub Pages 发布。

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

## 写一篇笔记

在 `src/content/posts` 新建 Markdown 文件：

```md
---
title: 笔记标题
description: 写下这条笔记以后为什么值得回来
publishDate: 2026-11-19
tags: [主题一, 主题二]
noteType: learning
readingStatus: finished
draft: false
featured: false
---

从这里开始记录理解、疑问和下一步。
```

`noteType` 支持：

- `paper`：论文阅读笔记；
- `learning`：课程、书籍、代码或方法的学习记录；
- `thought`：阶段性的思考与问题。

`readingStatus` 支持 `reading`、`finished` 和 `revisit`，分别对应阅读中、已读和待重读。

## 写一篇论文笔记

复制 `src/content/posts/_paper-note-template.md`，修改文件名并把 `draft` 改为 `false`。论文笔记可以额外填写：

```yaml
noteType: paper
readingStatus: reading
paperTitle: 论文原题
authors: [Author One, Author Two]
paperYear: 2026
venue: 会议或期刊
paperUrl: https://arxiv.org/abs/xxxx.xxxxx
```

模板建议保留“为什么读”“一句话结论”“核心方法”“我的理解与疑问”和“下次回看”等部分。最重要的不是复述摘要，而是记录自己的判断和没有想清楚的地方。

当前公开内容都设置了 `isExample: true`，用于展示笔记库、论文信息和正文排版。正式使用时可以删除或改写后移除该字段。

## 常用修改位置

| 需要修改的内容 | 文件 |
| --- | --- |
| 站点名称、描述、GitHub 地址 | `src/config.ts` |
| 首页介绍和版块文案 | `src/pages/index.astro` |
| 个人介绍与关注方向 | `src/pages/about.astro` |
| 颜色、字体与通用样式 | `src/styles/global.css` |
| 笔记字段规则 | `src/content.config.ts` |
| 搜索与筛选 | `src/pages/posts/index.astro` |

## 发布到 GitHub Pages

仓库已经包含 `.github/workflows/deploy.yml`。首次发布前：

1. 打开 GitHub 仓库的 **Settings → Pages**；
2. 在 **Build and deployment → Source** 中选择 **GitHub Actions**；
3. 将代码推送到 `main` 分支；
4. 等待 `Deploy to GitHub Pages` 工作流完成。

用户站点的地址是 <https://krisyang7.github.io>。这个仓库不需要设置 Astro 的 `base` 路径。

如果以后使用自定义域名，请在 `public/CNAME` 写入域名，并同步修改 `astro.config.mjs` 和 `src/config.ts` 中的网址。
