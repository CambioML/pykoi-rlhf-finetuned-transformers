import fs from "fs";
import path from "path";
import process from "process";

// read the directory
const assetsDirPath = path.resolve(process.cwd(), "dist", "assets");
const files = fs.readdirSync(assetsDirPath);

// find the JavaScript file
const jsFile = files.find((file) => file.endsWith(".js"));
if (!jsFile) {
  console.error("No JS file found");
  process.exit(1);
}

// read the JavaScript file
const jsFilePath = path.join(assetsDirPath, jsFile);
const jsContent = fs.readFileSync(jsFilePath, "utf-8");

// read the HTML file
const htmlFilePath = path.join(process.cwd(), "dist", "index.html");
let htmlContent = fs.readFileSync(htmlFilePath, "utf-8");

// inject the JS content into the HTML file
htmlContent = htmlContent.replace(
  "</body>",
  `<script>${jsContent}</script></body>`
);

// write the final output to a new HTML file
const outputFilePath = path.join(process.cwd(), "dist", "inline.html");
fs.writeFileSync(outputFilePath, htmlContent);

console.log(`Written to ${outputFilePath}`);
