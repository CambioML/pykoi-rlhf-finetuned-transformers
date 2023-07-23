<script>
  import { onMount } from "svelte";
  import { scaleLinear } from "d3-scale";
  import Cell from "./Cell.svelte";

  export let data;
  export let options;

  let maxCellChars = 25;

  const defaultOptions = { columns: {}, style: "normal", paged: 25 };
  options = { ...defaultOptions, ...options };

  const colorScale = scaleLinear()
    .domain([-1, 0, 1])
    .range(["#FF5470", "#f8f8f8", "#00ebc7"]);

  let { sortable, index, paged } = options;
  let sortKey = undefined;
  let sortDirection = true;
  let page = 0;
  if (sortable && index) {
    throw new Error("A table can either be ranked or sortable, but not both");
  }
  let columns = Object.keys(data[0]).map((key) => {
    const opts = options.columns[key] || {};
    return {
      key: key,
      type: opts.type || typeof data[0][key],
      options: opts,
    };
  });

  let rows = [];
  let pages = 1;
  index = 1;
  sortable = 1;
  $: {
    if (sortKey) {
      data = data.slice().sort((a, b) => {
        let as = a[sortKey];
        let bs = b[sortKey];
        if (as == bs) return JSON.stringify(a).localeCompare(JSON.stringify(b));
        let res = as > bs ? 1 : as < bs ? -1 : 0;
        if (sortDirection) res = -res;
        return res;
      });
    }
    let offset = page * paged;
    rows = data.slice(offset, offset + (paged || data.length));
    pages = paged ? Math.ceil(data.length / paged) : 1;
  }

  const identity = (value) => value; // Define your identity function here if needed

  let activeCell = null;
</script>

<!-- then, in your HTML -->
<div class="table">
  <div>
    <table class="pretty-table {options.style}">
      <thead>
        {#each columns as c}
          <th
            on:click={() => {
              if (sortable) {
                if (sortKey === c.key) {
                  sortDirection = !sortDirection;
                }
                sortKey = c.key;
              }
            }}>{c.key}</th
          >
        {/each}
      </thead>
      <tbody>
        {#each rows as row, rowIndex}
          <tr>
            {#each columns as c, columnIndex (c.key)}
              <td
                class:active={activeCell === `${rowIndex}-${columnIndex}`}
                class="cell-type-{c.type}"
                style="background: {c.key === 'change'
                  ? colorScale(row[c.key])
                  : 'none'}"
                on:click={() => {
                  activeCell =
                    activeCell === `${rowIndex}-${columnIndex}`
                      ? null
                      : `${rowIndex}-${columnIndex}`;
                }}
              >
                {row[c.key].length > maxCellChars &&
                activeCell !== `${rowIndex}-${columnIndex}`
                  ? `${row[c.key].substring(0, 40)}...`
                  : row[c.key]}
              </td>
            {/each}
          </tr>
        {/each}
      </tbody>
    </table>
    <!-- pagination -->
    {#if pages > 1}
      <div class="pretty-pager">
        <button on:click={() => (page = page > 0 ? page - 1 : 0)}
          >Previous</button
        >
        {#each Array(pages).fill() as _, i}
          <button on:click={() => (page = i)}>{i + 1}</button>
        {/each}
        <button
          on:click={() => (page = page < pages - 1 ? page + 1 : pages - 1)}
          >Next</button
        >
      </div>
    {/if}
  </div>
</div>

<style>
  @import url("https://fonts.googleapis.com/css?family=Work+Sans:400|Lato:400|Inconsolata:400");
  * {
    font-family: "Lato", monospace;
  }

  .table {
    margin: auto;
    width: 100%;
  }

  .pretty-pager {
    padding-top: 1rem;
  }
  .pretty-pager button {
    cursor: pointer;
    border-radius: 3px;
    border: 1px solid #fff;
    font-size: inherit;
  }
  .pretty-pager button:hover {
    border: 1px solid #888;
    color: red;
  }
  .pretty-table.normal {
    font-size: 15px;
  }
  .pretty-table.normal th,
  .pretty-table.normal td {
    padding: 3px 2px;
  }
  .pretty-table th,
  .pretty-table td {
    vertical-align: top;
  }
  .pretty-table thead th {
    text-transform: uppercase;
    font-weight: bold;
    font-family: "Work Sans", sans-serif;
    border-bottom: 2px solid black;
  }

  .pretty-table th {
    cursor: pointer;
  }
  .pretty-table tbody td.cell-type-number,
  .pretty-table tbody td.cell-rank {
    text-align: right;
  }
  .pretty-table tbody td.cell-type-number,
  .pretty-table tbody td.cell-rank {
    font-family: menlo, consolas, monaco, monospace;
    font-size: 90%;
  }
  .pretty-table tbody td.cell-rank {
    padding-right: 1em;
    color: #666;
  }
  table.pretty-table {
    border-collapse: collapse;
  }
  table.pretty-table {
    border-collapse: collapse;
    table-layout: fixed; /* Add this line */
  }
  .pretty-table tr {
    border-bottom: 1px solid #eee;
  }

  td.active {
    max-height: 100%;
    overflow: auto; /* Make it scrollable */
    color: red;
  }

  td:not(.active) {
    overflow: hidden;
    white-space: normal;
  }

  tr:hover {
    background: #eee;
  }

  .pretty-table {
    width: 100%;
  }
</style>
