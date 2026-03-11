// ── State ─────────────────────────────────────────────────────────────────────
let questions       = [];
let currentIndex    = 0;
let score           = 0;
let sessionId       = null;
let currentQuestion = null;
let selectedIndex   = null;

// ── Helpers ───────────────────────────────────────────────────────────────────
const show = (id) => document.getElementById(id).classList.remove('hidden');
const hide = (id) => document.getElementById(id).classList.add('hidden');
const el   = (id) => document.getElementById(id);

function showError(msg) {
  el('error-msg').textContent = msg;
  show('error-msg');
}

async function startQuiz() {
  const topic      = el('topic-select').value;
  const difficulty = el('difficulty-select').value;
  hide('setup-panel');
  hide('error-msg');
  show('loading');

  const res  = await fetch('/api/session/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ topic, difficulty })
  });
  const data = await res.json();
  sessionId  = data.session_id;

  questions = await loadQuestions(topic, difficulty);
  if (!questions.length) {
    showError('Could not generate questions. Please try again.');
    show('setup-panel');
    hide('loading');
    return;
  }

  hide('loading');
  currentIndex = 0;
  score        = 0;
  renderQuestion();
  show('quiz-panel');
}

async function loadQuestions(topic, difficulty) {
  const res = await fetch('/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ topic, difficulty, count: 5 })
  });
  if (!res.ok) { showError('Could not generate questions. Please try again.'); return []; }
  return await res.json();
}

function renderQuestion() {
  currentQuestion = questions[currentIndex];
  selectedIndex   = null;
  el('question-counter').textContent = `Question ${currentIndex + 1} of ${questions.length}`;
  el('question-text').textContent    = currentQuestion.question;
  el('progress-fill').style.width    = `${(currentIndex / questions.length) * 100}%`;

  const optsList = el('options-list');
  optsList.innerHTML = '';
  currentQuestion.options.forEach((opt, i) => {
    const btn       = document.createElement('button');
    btn.className   = 'option-btn';
    btn.textContent = opt;
    btn.onclick     = () => selectAnswer(i);
    optsList.appendChild(btn);
  });

  hide('feedback');
  hide('ai-explanation');
  hide('ai-explain-btn');
  el('ai-explanation').textContent = '';
}

function selectAnswer(index) {
  if (selectedIndex !== null) return;
  selectedIndex = index;
  const isCorrect = index === currentQuestion.answerIndex;
  if (isCorrect) score++;

  const btns = el('options-list').querySelectorAll('.option-btn');
  btns.forEach(b => b.disabled = true);
  btns[currentQuestion.answerIndex].classList.add('correct');
  if (!isCorrect) btns[index].classList.add('wrong');

  el('feedback-text').textContent    = isCorrect ? '✓ Correct!' : '✗ Incorrect.';
  el('explanation-text').textContent = currentQuestion.explanation || '';
  show('feedback');

  if (!isCorrect) {
    show('ai-explain-btn');
    el('ai-explain-btn').disabled    = false;
    el('ai-explain-btn').textContent = 'Ask AI to Explain';
  }

  recordAnswer(currentQuestion, index, isCorrect);
}

function recordAnswer(question, selected, isCorrect) {
  fetch('/api/session/answer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id:  sessionId,
      question_id: question.id,
      topic:       question.topic,
      selected:    selected,
      correct:     isCorrect ? 1 : 0
    })
  });
}

function nextQuestion() {
  currentIndex++;
  if (currentIndex < questions.length) { renderQuestion(); } else { showResults(); }
}

function showResults() {
  hide('quiz-panel');
  el('score-text').textContent = `You scored ${score} out of ${questions.length}.`;
  show('results-panel');
  hide('recommendation-box');
}

async function fetchExplanation() {
  const btn = el('ai-explain-btn');
  btn.textContent = 'Thinking...';
  btn.disabled    = true;
  try {
    const res  = await fetch('/api/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question:       currentQuestion.question,
        options:        currentQuestion.options,
        answer_index:   currentQuestion.answerIndex,
        selected_index: selectedIndex
      })
    });
    const data = await res.json();
    el('ai-explanation').textContent = data.explanation;
  } catch (e) {
    el('ai-explanation').textContent = 'Explanation unavailable. Please try again.';
  }
  show('ai-explanation');
  hide('ai-explain-btn');
}

async function fetchRecommendation() {
  const btn = el('recommend-btn');
  btn.textContent = 'Thinking...';
  btn.disabled    = true;
  try {
    const res  = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId })
    });
    const data = await res.json();
    el('recommendation-box').textContent = data.recommendation;
  } catch (e) {
    el('recommendation-box').textContent = 'Recommendation unavailable. Please try again.';
  }
  show('recommendation-box');
}

function resetQuiz() {
  hide('results-panel');
  hide('recommendation-box');
  show('setup-panel');
  sessionId = null; questions = []; currentIndex = 0; score = 0;
}

// ── Student ID footer ─────────────────────────────────────────────────────────
(async function loadStudentId() {
  try {
    const res  = await fetch('/api/info');
    const data = await res.json();
    document.getElementById('student-id-display').textContent = data.student_id;
    document.getElementById('student-id-top').textContent = data.student_id;
  } catch (e) {
    document.getElementById('student-id-display').textContent = 'unavailable';
    document.getElementById('student-id-top').textContent = 'unavailable';
  }
})();
