import { defineConfig } from "astro/config";
import fs from "fs";
import mdx from "@astrojs/mdx";
import tailwind from "@astrojs/tailwind";
import sitemap from "@astrojs/sitemap";
import prefetch from "@astrojs/prefetch";
import remarkUnwrapImages from "remark-unwrap-images";

// https://astro.build/config
export default defineConfig({
  // ! Please remember to replace the following site property with your own domain
  site: "https://yxiang.dev/",
  markdown: {
    remarkPlugins: [remarkUnwrapImages],
    remarkRehype: {
      footnoteLabelProperties: {
        className: [""],
      },
    },
    shikiConfig: {
      theme: "one-dark-pro",
      wrap: false,
    },
  },
  integrations: [
    mdx({}),
    tailwind({
      applyBaseStyles: false,
    }),
    sitemap(),
    prefetch(),
  ],
  vite: {
    plugins: [rawFonts([".ttf"])],
    optimizeDeps: {
      exclude: ["@resvg/resvg-js"],
    },
  },
});
function rawFonts(ext: Array<string>) {
  return {
    name: "vite-plugin-raw-fonts",
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore:next-line
    transform(_, id) {
      if (ext.some((e) => id.endsWith(e))) {
        const buffer = fs.readFileSync(id);
        return {
          code: `export default ${JSON.stringify(buffer)}`,
          map: null,
        };
      }
    },
  };
}
