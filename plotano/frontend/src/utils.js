export function isSvelteStore(object) {
  return object && typeof object.subscribe === "function";
}
