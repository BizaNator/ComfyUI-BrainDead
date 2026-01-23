const path = require('path');
const TerserPlugin = require('terser-webpack-plugin');

module.exports = {
    entry: './entry.js',
    output: {
        filename: 'three-inspector-bundle.js',
        path: path.resolve(__dirname, '../web/js'),
    },
    mode: 'production',
    optimization: {
        minimize: true,
        minimizer: [new TerserPlugin({
            terserOptions: {
                format: { comments: false },
            },
            extractComments: {
                condition: /^\**!|@license|@preserve/i,
                filename: (fileData) => `${fileData.filename}.LICENSE.txt`,
            },
        })],
    },
    performance: {
        hints: false,
    },
};
