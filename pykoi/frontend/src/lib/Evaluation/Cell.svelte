<!-- Cell.svelte -->
<script>
  export let content;
  export let isActive;
</script>

<div class="cell" class:active={isActive}>
  <div class="content">
    {isActive
      ? content
      : content.length > 40
      ? `${content.substring(0, 40)}...`
      : content}
  </div>
</div>

<style>
  .cell {
    max-height: 40px;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }

  .cell.active {
    max-height: 100%;
    overflow: visible;
    white-space: normal;
  }
</style>
