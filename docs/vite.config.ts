import {SearchPlugin} from "vitepress-plugin-search";
import {defineConfig} from "vite";

//default options
const options = {
    previewLength: 62,
    buttonLabel: "Search",
    placeholder: "Search docs",
};

export default defineConfig({
    // @ts-ignore
    plugins: [SearchPlugin(options)],
});