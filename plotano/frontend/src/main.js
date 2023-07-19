import "./app.css";
import "./normalize.css";
import "./styles.css";
import App from "./App.svelte";

const app = new App({
  target: document.getElementById("app"),
});

export default app;
