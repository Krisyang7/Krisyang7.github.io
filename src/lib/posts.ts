import { getCollection, type CollectionEntry } from 'astro:content';

export type Post = CollectionEntry<'posts'>;

export async function getPublishedPosts() {
  const posts = await getCollection('posts', ({ data }) => !data.draft);
  return posts.sort(
    (a, b) => b.data.publishDate.valueOf() - a.data.publishDate.valueOf(),
  );
}

export function formatDate(date: Date) {
  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  }).format(date);
}

export function getReadingTime(body = '') {
  const plainText = body.replace(/```[\s\S]*?```|<[^>]+>|[#>*_`\[\]()-]/g, ' ');
  const cjkCount = (plainText.match(/[\u3400-\u9fff]/g) ?? []).length;
  const wordCount = plainText
    .replace(/[\u3400-\u9fff]/g, ' ')
    .trim()
    .split(/\s+/)
    .filter(Boolean).length;
  return Math.max(1, Math.ceil((cjkCount + wordCount * 2) / 400));
}

export function postUrl(post: Post) {
  return `/posts/${post.id}/`;
}
