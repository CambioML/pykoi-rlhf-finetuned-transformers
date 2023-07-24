<script>
  import HorizontalStackedBar from "./DashboardFeedback/HorizontalStackedBar.svelte";
  import QATable from "./DashboardFeedback/QATable.svelte";
  import ColumnChart from "./DashboardFeedback/ColumnChart.svelte";
  import HistogramAnswer from "./DashboardFeedback/HistogramAnswer.svelte";
  import HistogramQuestion from "./DashboardFeedback/HistogramQuestion.svelte";
  import MetricCardAbsolute from "./DashboardFeedback/MetricCardAbsolute.svelte";
  import MetricCardPercentage from "./DashboardFeedback/MetricCardPercentage.svelte";

  import { onMount } from "svelte";
  import { chatLog } from "../../store";

  onMount(() => {
    getDataFromDB();
  });

  async function getDataFromDB() {
    const response = await fetch("/chat/qa_table/retrieve");
    const data = await response.json();

    const dbRows = data["rows"];
    const formattedRows = dbRows.map((row) => ({
      id: row[0],
      question: row[1],
      answer: row[2],
      vote_status: row[3],
    }));
    $chatLog = [...formattedRows];
  }
</script>

<div class="feedback-container">
  <div class="feedback-left">
    <div class="left-text">
      <div class="card-1">
        <MetricCardAbsolute />
      </div>
      <div class="card-2">
        <MetricCardPercentage />
      </div>
    </div>
    <div class="left-confidence">
      <HorizontalStackedBar />
    </div>
    <div class="left-question">
      <ColumnChart feedback={"Bad"} />
    </div>
    <div class="left-filter">
      <HistogramQuestion />
    </div>
    <div class="left-filter2">
      <HistogramAnswer />
    </div>
  </div>
  <div class="feedback-right">
    <div class="right-chart">
      <div class="right-chart-1" />
      <div class="right-chart-2" />
    </div>
    <div class="right-table">
      <QATable />
    </div>
  </div>
</div>

<style>
  .feedback-container {
    border-bottom: var(--line);
    border-right: var(--line);
    display: grid;
    grid-template-columns: 30% 70%;
    grid-template-rows: 100%;
    width: 90%;
    height: calc(100vh - var(--headerHeight));
  }
  .feedback-left {
    border-bottom: var(--line);
    display: grid;
    grid-template-columns: 100%;
    grid-template-rows: 25% 25% 25% 12.5% 12.5%;
  }
  .left-text {
    text-align: center;
    border-bottom: var(--line);
    border-right: var(--line);
    display: grid;
    grid-template-rows: 100%;
    grid-template-columns: 50% 50%;
  }
  .left-confidence {
    text-align: center;
    border-bottom: var(--line);
    border-right: var(--line);
    display: grid;
    grid-template-rows: 40% 55%;
  }
  .left-question {
    border-bottom: var(--line);
    border-right: var(--line);
  }
  .left-filter {
    border-bottom: var(--line);
    border-right: var(--line);
  }
  .feedback-right {
    border-bottom: var(--line);
    border-right: var(--line);
    display: grid;
    grid-template-columns: 100%;
    grid-template-rows: 30% 70%;
    grid-template-rows: 0% 100%;
  }
  .right-chart {
    border-bottom: var(--line);
    border-right: var(--line);
    display: grid;
    grid-template-columns: 50% 50%;
    grid-template-rows: 100%;
  }
  .right-chart-1,
  .right-chart-2 {
    border-right: var(--line);
  }
  .right-table {
    border-bottom: var(--line);
    border-right: var(--line);
  }

  .card-1 {
    border-right: var(--line);
  }
</style>
