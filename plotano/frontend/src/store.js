import { writable, derived } from "svelte/store";
import { tweened } from "svelte/motion";

export const state = writable({});
export const chatLog = writable([]);
export const rankChatLog = writable([]);

// feedback (rewrite)
export const feedbackSelection = writable("down"); // up, down, or NA

export const examples = writable([
  {
    index: 1,
    question: "What is the capital of France?",
    answer:
      "The capital of France is Paris.The capital of France is Paris.The capital of France is Paris.",
    vote: "down",
  },
  {
    index: 2,
    question: "Who was the first president of the United States?",
    answer: "The first president of the United States was George Washington.",
    vote: "up",
  },
  {
    index: 3,
    question: "What is the formula for water?",
    answer: "The formula for water is H2O.",
    vote: "n/a",
  },
  {
    index: 4,
    question: "Who wrote the book '1984'?",
    answer:
      "The book '1984' was written by George Orwell '1984' was written by George Orwell '1984' was written by George Orwell  v.",
    vote: "up",
  },
  {
    index: 5,
    question: "Who discovered penicillin?",
    answer: "Penicillin was discovered by Alexander Fleming.",
    vote: "up",
  },
  {
    index: 6,
    question: "Who painted the Mona Lisa?",
    answer: "The Mona Lisa was painted by Leonardo da Vinci.",
    vote: "down",
  },
  {
    index: 7,
    question: "What is the square root of 64?",
    answer: "The square root of 64 is 8.",
    vote: "n/a",
  },
  {
    index: 8,
    question: "What is the distance from the Earth to the Moon?",
    answer:
      "The distance from the Earth to the Moon is approximately 238,855 miles.",
    vote: "down",
  },
  {
    index: 9,
    question: "What is the capital of Canada?",
    answer: "The capital of Canada is Ottawa.",
    vote: "n/a",
  },
  {
    index: 10,
    question: "Why is the current president of the United States?",
    answer: "The current president of the United States is Joe Biden.",
    vote: "n/a",
  },
  {
    index: 11,
    question: "Can is the theory of relativity?",
    answer:
      "The theory of relativity, proposed by Albert Einstein, describes the laws of physics especially in relation to gravitation and high velocities. It consists of two parts: the special theory of relativity, which pertains to motion in the absence of gravitational fields, and the general theory of relativity, a theory of gravitation that is based on the principle of equivalence, in which gravitational and inertial forces are regarded as locally indistinguishable.",
    vote: "up",
  },
  {
    index: 12,
    question: "What is global warming?",
    answer:
      "Global warming is the long-term heating of Earth's climate system observed since the pre-industrial period (between 1850 and 1900) due to human activities, primarily fossil fuel burning, which increases heat-trapping greenhouse gas levels in Earth's atmosphere. The term is frequently used interchangeably with the term climate change.",
    vote: "up",
  },
  {
    index: 13,
    question: "Where was Winston Churchill?",
    answer:
      "Sir Winston Leonard Spencer-Churchill was a British statesman and military leader who served as Prime Minister of the United Kingdom from 1940 to 1945, during the Second World War, and again from 1951 to 1955. His leadership during World War II played a key role in the Allied victory over the Axis Powers.",
    vote: "n/a",
  },
  {
    index: 14,
    question: "Where is the Great Barrier Reef?",
    answer:
      "The Great Barrier Reef is the world's largest coral reef system composed of over 2,900 individual reefs and 900 islands stretching for over 2,300 kilometres over an area of approximately 344,400 square kilometres. It is located in the Coral Sea, off the coast of Queensland, Australia and is renowned for its diverse marine life and its immense natural beauty.",
    vote: "up",
  },
  {
    index: 15,
    question: "Where is the purpose of the United Nations?",
    answer:
      "The United Nations is an international organization founded in 1945 after the Second World War by 51 countries committed to maintaining international peace and security, developing friendly relations among nations and promoting social progress, better living standards and human rights. It provides a forum for its member states to express their views, through the General Assembly, the Security Council, the Economic and Social Council and other bodies and committees.",
    vote: "up",
  },
  {
    index: 16,
    question: "What are black holes?",
    answer:
      "Black holes are regions of spacetime where gravity is so strong that nothing, not even particles and electromagnetic radiation such as light, can escape its pull. They are the result of the dying remnants of massive stars. Because it is impossible to observe black holes directly, they are observed by the gravitational effects they have on the matter around them.",
    vote: "down",
  },
  {
    index: 17,
    question: "What are human rights?",
    answer:
      "Human rights are basic rights and freedoms that belong to every person in the world, from birth until death. These basic rights include the right to life and liberty, freedom from slavery and torture, freedom of opinion and expression, the right to work and education, and many more. They are fundamental to living a life of dignity and are protected internationally by law, treaties, customary international law, general principles and other sources of international law.",
    vote: "n/a",
  },
  {
    index: 18,
    question: "What is democracy?",
    answer:
      "Democracy is a form of government in which power is vested in the people, who rule either directly or through freely elected representatives. Democracy is characterized by elements such as freedom of speech, inclusiveness, equality, consent of the governed, and emphasis on individual freedom. It is the opposite of a dictatorship.",
    vote: "down",
  },
  {
    index: 19,
    question: "Where was William Shakespeare?",
    answer:
      "William Shakespeare was an English poet, playwright, and actor, widely regarded as the greatest writer in the English language and the world's greatest dramatist. He is often called England's national poet and the 'Bard of Avon'. His works, including collaborations, consist of approximately 39 plays, 154 sonnets, two long narrative poems, and a few other verses, some of uncertain authorship.",
    vote: "down",
  },
  {
    index: 20,
    question: "What is photosynthesis?",
    answer:
      "Photosynthesis is a process used by plants and other organisms to convert light energy, usually from the Sun, into chemical energy that can be later released to fuel the organisms' activities. This chemical energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water. Photosynthesis is fundamentally important for life on Earth as it provides the oxygen that all complex life needs to survive.",
    vote: "n/a",
  },
]);

const initialTally = [{ "n/a": 1, up: 1, down: 1 }];
export const stackedData = writable({ "n/a": 1, up: 1, down: 1 });

// export const stackedData = derived(
//   chatLog,
//   ($chatLog) => {
//     let newTally = { ...initialTally[0] };
//     $chatLog.forEach((example) => {
//       newTally[example.vote]++;
//     });
//     console.log("newTally", newTally);
//     return [{ ...newTally }];
//   },
//   initialTally
// );

const questions = ["Who", "What", "How", "Why", "Where", "Does", "Can", "N/A"];

export const questionDistribution = tweened(
  questions.map((question) => ({ question, count: 0 }))
);
