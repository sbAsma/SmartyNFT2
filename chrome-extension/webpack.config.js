const path = require('path');
const CopyPlugin = require('copy-webpack-plugin');

module.exports = {
    mode: "production",
    entry: {
        "content-script": path.resolve(__dirname, 'src', 'content-script.js'),
    },
    output: {
        path: path.join(__dirname, "dist"),
        filename: "[name].js",
    },
    resolve: {
        extensions: [".js"],
    },
    plugins: [
        new CopyPlugin({
            patterns: [{from: "./src", to: ".", }]
        }),
    ],
};
