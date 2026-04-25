/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Ensure knowledge.json and welsh-terms.json are bundled with API serverless functions.
  // Next.js 14.x uses the experimental namespace for this option.
  experimental: {
    outputFileTracingIncludes: {
      '/api/**': ['./knowledge.json', './welsh-terms.json'],
    },
  },
};

module.exports = nextConfig;
