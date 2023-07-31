export function isSvelteStore(object) {
  return object && typeof object.subscribe === "function";
}

export function formatBytes(bytes, decimals = 2) {
  if (bytes === 0) return "0 Bytes";

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ["Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
}

export function tallyQuestions(data) {
  // Create an empty object to store the count of each question category
  let questionCounts = {
    who: 0,
    what: 0,
    how: 0,
    why: 0,
    where: 0,
    does: 0,
    can: 0,
    "n/a": 0,
  };

  // Iterate through each item in the data
  for (let i = 0; i < data.length; i++) {
    // Get the first word of the question (i.e., the question category)
    let questionCategory = data[i].question.split(" ")[0].toLowerCase();

    // If this question category exists in our counts object, increment its count
    if (questionCategory in questionCounts) {
      questionCounts[questionCategory]++;
    }
    // Otherwise, count it as "n/a"
    else {
      questionCounts["n/a"]++;
    }
  }

  // Transform the counts object into an array of objects
  let result = Object.keys(questionCounts).map((key) => ({
    question: key,
    count: questionCounts[key],
  }));

  return result;
}

export function getQAWordFrequency(arr) {
  return arr.map((item) => {
    const questionTokens = item.question.split(" ").length;
    const answerTokens = item.answer.split(" ").length;

    return {
      index: item.index,
      question: +questionTokens,
      answer: +answerTokens,
      vote: item.vote,
    };
  });
}
