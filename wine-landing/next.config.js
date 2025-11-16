/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  distDir: "out",

  basePath: "/wine-quality-forecasting",
  assetPrefix: "/wine-quality-forecasting/",

  images: {
    unoptimized: true,
  },
};

module.exports = nextConfig;
