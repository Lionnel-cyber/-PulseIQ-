import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  turbopack: {
    root: __dirname,
  },
  transpilePackages: [
    "@deck.gl/react",
    "@deck.gl/core",
    "@deck.gl/layers",
    "@deck.gl/geo-layers",
    "react-map-gl",
  ],
};

export default nextConfig;
