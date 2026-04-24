"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var prism_react_renderer_1 = require("prism-react-renderer");
var math = require("remark-math");
var katex = require("rehype-katex");
var config = {
    markdown: {
        mermaid: true,
        hooks: {
            onBrokenMarkdownLinks: "warn",
        },
    },
    themes: ["@docusaurus/theme-mermaid"],
    title: "Learn AI, ML",
    tagline: "Learn AI & ML from Zero to Hero",
    favicon: "img/favicon.ico",
    url: "https://your-docusaurus-site.example.com",
    baseUrl: "/",
    organizationName: "facebook",
    projectName: "docusaurus",
    onBrokenLinks: "throw",
    i18n: {
        defaultLocale: "en",
        locales: ["en"],
    },
    presets: [
        [
            "classic",
            {
                docs: {
                    sidebarPath: "./sidebars.ts",
                    remarkPlugins: [math],
                    rehypePlugins: [katex],
                },
                theme: {
                    customCss: "./src/css/custom.css",
                },
            },
        ],
    ],
    plugins: [
        [
            require.resolve("@easyops-cn/docusaurus-search-local"),
            {
                hashed: true,
                indexDocs: true,
                indexBlog: true,
                indexPages: true,
                docsRouteBasePath: "/",
                searchResultContextMaxLength: 80,
            },
        ],
    ],
    stylesheets: [
        {
            href: "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css",
            type: "text/css",
            integrity: "sha384-nW7hDJ4SmCyFRIE7NRH8fXf2l+dNwM4hxKyn9YOeAWMq8TDzv2h7Q+n9AiUMav+s",
            crossorigin: "anonymous",
        },
    ],
    themeConfig: {
        image: "img/docusaurus-social-card.jpg",
        /** ✅ Correct Mermaid config (no themeVariables — use CSS override instead) **/
        mermaid: {
            theme: {
                light: "neutral",
                dark: "dark",
            },
        },
        navbar: {
            title: "AI & ML",
            logo: {
                alt: "AI ML Logo",
                src: "img/logo.svg",
            },
            items: [
                {
                    type: "docSidebar",
                    sidebarId: "tutorialSidebar",
                    position: "left",
                    label: "Tutorial",
                },
                { type: "search", position: "right" },
            ],
        },
        footer: {
            style: "dark",
            links: [
                {
                    title: "Docs",
                    items: [{ label: "Tutorial", to: "/docs/intro" }],
                },
                {
                    title: "Community",
                    items: [{ label: "ChatGPT", href: "https://chatgpt.com/" }],
                },
            ],
            copyright: "Copyright \u00A9 ".concat(new Date().getFullYear(), " Learn AI, ML. Built with Docusaurus."),
        },
        prism: {
            theme: prism_react_renderer_1.themes.github,
            darkTheme: prism_react_renderer_1.themes.dracula,
        },
    },
};
exports.default = config;
