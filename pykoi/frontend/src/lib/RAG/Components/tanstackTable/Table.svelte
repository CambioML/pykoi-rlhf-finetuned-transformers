<script>
    import {
      createColumnHelper,
      flexRender,
      getCoreRowModel,
      createSvelteTable,
      getSortedRowModel,
      getFilteredRowModel,
      getFacetedRowModel,
      getPaginationRowModel,
    } from "@tanstack/svelte-table";
    import { rankItem } from "@tanstack/match-sorter-utils";
    import { writable } from "svelte/store";
    import { uploadedFiles } from "../../../store.js";
    import { formatBytes } from "../../../../utils.js";

    //   export let data = [
    //     { file: "jared", size: 2400, type: "md" },

    //   ];

    function getSortSymbol(isSorted) {
      return isSorted ? (isSorted === "asc" ? "🔼" : "🔽") : "";
    }

    const globalFilterFn = (row, columnId, value, addMeta) => {
      console.log("yessir");
      if (Array.isArray(value)) {
        if (value.length === 0) return true;
        return value.includes(row.getValue(columnId));
      }
      if (typeof value === "number") value = String(value);

      // Rank the item
      const itemRank = rankItem(row.getValue(columnId), value);

      // Store the itemRank info
      addMeta({
        itemRank,
      });

      // Return if the item should be filtered in/ou

      return itemRank.passed;
    };

    const columnHelper = createColumnHelper();

    const columns = [
      columnHelper.accessor("file", {
        header: "File",
        cell: (info) => info.getValue(),
        footer: (info) => info.column.id,
      }),
      columnHelper.accessor("size", {
        header: "Size",
        cell: (info) => formatBytes(info.getValue()),
        footer: (info) => info.column.id,
      }),
      columnHelper.accessor("type", {
        header: "Type",
        cell: (info) => info.getValue(),
        footer: (info) => info.column.id,
      }),
    ];

    let globalFilter = "";

    let options = writable({
      data: $uploadedFiles,
      columns: columns,
      getCoreRowModel: getCoreRowModel(),
      getSortedRowModel: getSortedRowModel(),
      getFilteredRowModel: getFilteredRowModel(),
      globalFilterFn: globalFilterFn,
      getFacetedRowModel: getFacetedRowModel(),
      getPaginationRowModel: getPaginationRowModel(),
      state: {
        globalFilter,
        pagination: {
          pageSize: 7,
          pageIndex: 0,
        },
      },
      enableGlobalFilter: true,
    });

    function setGlobalFilter(filter) {
      globalFilter = filter;
      options.update((old) => {
        return {
          ...old,
          state: {
            ...old.state,
            globalFilter: filter,
          },
        };
      });
    }

    function setPageSize(e) {
      const target = e.target;
      options.update((old) => {
        return {
          ...old,
          state: {
            ...old.state,
            pagination: {
              ...old.state?.pagination,
              pageSize: parseInt(target.value),
            },
          },
        };
      });
    }

    function setCurrentPage(page) {
      options.update((old) => {
        return {
          ...old,
          state: {
            ...old.state,
            pagination: {
              ...old.state?.pagination,
              pageIndex: page,
            },
          },
        };
      });
    }

    let timer;
    function handleSearch(e) {
      clearTimeout(timer);
      timer = setTimeout(() => {
        const target = e.target;
        setGlobalFilter(target.value);
      }, 100);
    }

    function handleCurrPageInput(e) {
      const target = e.target;
      setCurrentPage(parseInt(target.value) - 1);
    }

    let table = createSvelteTable(options);

    let headerGroups = $table.getHeaderGroups();
    //   let options;
    //   make sure options and table update whenever uploadedFiles does.
    $: {
      options = {
        data: $uploadedFiles,
        columns: columns,
        getCoreRowModel: getCoreRowModel(),
        getSortedRowModel: getSortedRowModel(),
        getFilteredRowModel: getFilteredRowModel(),
        globalFilterFn: globalFilterFn,
        getFacetedRowModel: getFacetedRowModel(),
        getPaginationRowModel: getPaginationRowModel(),
        state: {
          globalFilter,
          pagination: {
            pageSize: 7,
            pageIndex: 0,
          },
        },
        enableGlobalFilter: true,
      };

      // Re-create the table whenever options changes
      table = createSvelteTable(writable(options));
    }
  </script>

  <div>
    <input
      type="search"
      class="input"
      on:keyup={handleSearch}
      on:search={handleSearch}
      placeholder="Search..."
    />
    <div class="table-container">
      <table class="table">
        <thead>
          {#each headerGroups as headerGroup}
            <tr>
              {#each headerGroup.headers as header}
                <th colSpan={header.colSpan}>
                  {#if !header.isPlaceholder}
                    <button
                      class:is-disabled={!header.column.getCanSort()}
                      disabled={!header.column.getCanSort()}
                      on:click={header.column.getToggleSortingHandler()}
                    >
                      <svelte:component
                        this={flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                      />
                      <span>
                        {getSortSymbol(header.column.getIsSorted())}
                      </span>
                    </button>
                  {/if}
                </th>
              {/each}
            </tr>
          {/each}
        </thead>
        <tbody>
          {#each $table.getRowModel().rows as row}
            <tr>
              {#each row.getVisibleCells() as cell (cell.id)}
                <td>
                  <svelte:component
                    this={flexRender(
                      cell.column.columnDef.cell,
                      cell.getContext()
                    )}
                  />
                </td>
              {/each}
            </tr>
          {/each}
        </tbody>
        <div class="is-flex is-align-items-center">
          <button
            class="button is-white"
            on:click={() =>
              setCurrentPage($table.getState().pagination.pageIndex - 1)}
            class:is-disabled={!$table.getCanPreviousPage()}
            disabled={!$table.getCanPreviousPage()}
          >
            {"<"}
          </button>
          <span> Page </span>
          <input
            type="number"
            value={$table.getState().pagination.pageIndex + 1}
            min={0}
            max={$table.getPageCount() - 1}
            on:change={handleCurrPageInput}
            class="mx-1"
          />
          <span>
            {" "}of{" "}
            {$table.getPageCount()}
          </span>
          <button
            class="button is-white"
            on:click={() =>
              setCurrentPage($table.getState().pagination.pageIndex + 1)}
            class:is-disabled={!$table.getCanNextPage()}
            disabled={!$table.getCanNextPage()}
          >
            {">"}
          </button>
        </div>
      </table>
    </div>
  </div>

  <style>
    .table-container {
      margin: auto;
      width: 100%;
      height: 100%;
      overflow-y: scroll;
      max-height: 100%; /* Set maximum height to 100% of its parent */
      height: 100%; /* Set height to 100% of its parent */
      overflow-y: auto;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: var(--smallText);
    }

    thead th {
      text-align: center;
      padding: 5px;
      border-bottom: 4px solid var(--grey);
      background-color: var(--lightGrey);
    }

    thead th:nth-child(1) {
      text-align: left;
    }

    tbody tr {
      border-bottom: var(--line);
    }

    tbody tr:nth-child(even) {
      background-color: var(--white);
    }

    tbody td {
      padding: 10px;
    }

    .is-flex {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px;
    }

    .button {
      padding: 5px 10px;
      border: none;
      cursor: pointer;
      transition: background-color 0.3s ease;
    }

    .button.is-disabled {
      cursor: not-allowed;
      color: #ccc;
    }

    .button:not(.is-disabled):hover {
      background-color: #f2f2f2;
    }

    .button.is-white {
      color: #000;
    }

    .mx-1 {
      margin-left: 1em;
      margin-right: 1em;
    }
  </style>