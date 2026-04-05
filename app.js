import { AutoModel, AutoTokenizer, mean_pooling } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js';

const MODEL_ID = 'onnx-community/embeddinggemma-300m-ONNX';

let chunks, embeddings, DIMS;
let embedder, tokenizer;

async function loadData() {
  const [embBuf, chunksData, metaData] = await Promise.all([
    fetch('data/embeddings.bin').then(r => {
      if (!r.ok) throw new Error('embeddings.bin not found. Run build_index.py first.');
      return r.arrayBuffer();
    }),
    fetch('data/chunks.json').then(r => {
      if (!r.ok) throw new Error('chunks.json not found. Run build_index.py first.');
      return r.json();
    }),
    fetch('data/meta.json').then(r => {
      if (!r.ok) throw new Error('meta.json not found. Run build_index.py first.');
      return r.json();
    }),
  ]);
  embeddings = new Float32Array(embBuf);
  chunks = chunksData;
  DIMS = metaData.dims;
}

function getVec(i) {
  return embeddings.subarray(i * DIMS, (i + 1) * DIMS);
}

async function loadEmbedder() {
  [embedder, tokenizer] = await Promise.all([
    AutoModel.from_pretrained(MODEL_ID, { dtype: 'q8' }),
    AutoTokenizer.from_pretrained(MODEL_ID),
  ]);
}

async function embedQuery(text) {
  const formatted = `task: question answering | query: ${text}`;
  const inputs = await tokenizer(formatted, {
    padding: true,
    truncation: true,
    max_length: 2048,
  });
  const output = await embedder(inputs);
  const pooled = mean_pooling(output.last_hidden_state, inputs.attention_mask);
  
  // Manual L2 normalization because Transformers.js pooled might not be normalized
  const norm = Math.hypot(...pooled.data) + 1e-10;
  return Float32Array.from(pooled.data, v => v / norm);
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function search(queryVec, k = 5) {
  const scores = [];
  for (let i = 0; i < chunks.length; i++) {
    scores.push({ i, score: dot(queryVec, getVec(i)) });
  }
  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, k).map(({ i, score }) => ({ ...chunks[i], score }));
}

async function synthesize(query, topChunks) {
  if (typeof LanguageModel === 'undefined') return null;
  
  let avail;
  try {
    avail = await LanguageModel.availability();
  } catch (e) {
    console.error('Error checking LanguageModel availability:', e);
    return null;
  }

  if (avail === 'unavailable') return null;

  const context = topChunks
    .map((c, i) => `[${i + 1}] ${c.title}\n${c.body}`)
    .join('\n\n---\n\n');

  const session = await LanguageModel.create({
    systemPrompt:
      'Answer the question using only the provided notes. Be concise. ' +
      'If the notes do not contain the answer, say so.',
  });

  const prompt = `Notes:\n${context}\n\nQuestion: ${query}`;
  return session.promptStreaming(prompt);
}

function loadConfig() {
  document.getElementById('llm-endpoint').value = localStorage.getItem('llm_endpoint') || '';
  document.getElementById('llm-model').value = localStorage.getItem('llm_model') || '';
  document.getElementById('llm-endpoint').addEventListener('change', e => {
    localStorage.setItem('llm_endpoint', e.target.value.trim());
  });
  document.getElementById('llm-model').addEventListener('change', e => {
    localStorage.setItem('llm_model', e.target.value.trim());
  });
}

async function synthesizeOpenAI(query, topChunks) {
  const endpoint = localStorage.getItem('llm_endpoint') || '';
  if (!endpoint) return null;
  const model = localStorage.getItem('llm_model') || 'default';

  const context = topChunks
    .map((c, i) => `[${i + 1}] ${c.title}\n${c.body}`)
    .join('\n\n---\n\n');

  const resp = await fetch(endpoint.replace(/\/$/, '') + '/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      stream: true,
      messages: [
        {
          role: 'system',
          content: 'Answer the question using only the provided notes. Be concise. If the notes do not contain the answer, say so.',
        },
        {
          role: 'user',
          content: `Notes:\n${context}\n\nQuestion: ${query}`,
        },
      ],
    }),
  });

  if (!resp.ok) throw new Error(`LLM endpoint returned ${resp.status}`);
  return resp.body;
}

async function renderOpenAIStream(body, targetEl) {
  targetEl.textContent = '';
  const reader = body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += value;
    const lines = buf.split('\n');
    buf = lines.pop();
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6).trim();
      if (data === '[DONE]') return;
      try {
        const delta = JSON.parse(data).choices?.[0]?.delta?.content;
        if (delta) targetEl.textContent += delta;
      } catch {}
    }
  }
}

async function renderStreamingAnswer(stream, targetEl) {
  targetEl.textContent = '';
  const reader = stream.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      targetEl.textContent += value;
    }
  } catch (e) {
    targetEl.textContent = 'Error during streaming: ' + e.message;
  }
}

function renderChunks(results) {
  const container = document.getElementById('results');
  container.innerHTML = '';
  results.forEach(res => {
    const card = document.createElement('div');
    card.className = 'result-card';
    card.innerHTML = `
      <span class="score">${res.score.toFixed(4)}</span>
      <h3>${res.title}</h3>
      <div class="tags">${res.tags.map(t => '#' + t).join(' ')}</div>
      <p>${res.body.substring(0, 200)}${res.body.length > 200 ? '...' : ''}</p>
      <span class="path">${res.path}</span>
    `;
    container.appendChild(card);
  });
}

function setStatus(msg) {
  document.getElementById('status').textContent = msg;
}

async function handleQuery(e) {
  e.preventDefault();
  const queryText = document.getElementById('query').value.trim();
  if (!queryText) return;

  setStatus('Embedding query…');
  const queryVec = await embedQuery(queryText);

  setStatus('Searching…');
  const results = search(queryVec, 5);
  renderChunks(results);

  const answerEl = document.getElementById('answer');
  const answerSection = document.getElementById('answer-section');
  
  try {
    const chromeStream = await synthesize(queryText, results);
    if (chromeStream) {
      setStatus('');
      answerSection.style.display = 'block';
      await renderStreamingAnswer(chromeStream, answerEl);
      return;
    }

    const llmBody = await synthesizeOpenAI(queryText, results);
    if (llmBody) {
      setStatus('');
      answerSection.style.display = 'block';
      await renderOpenAIStream(llmBody, answerEl);
      return;
    }

    setStatus('No AI backend available — showing retrieved notes only.');
    answerSection.style.display = 'none';
  } catch (err) {
    console.error(err);
    setStatus('Error during synthesis: ' + err.message);
  }
}

async function init() {
  loadConfig();
  try {
    await loadData();
    setStatus('Loading model...');
    await loadEmbedder();
    setStatus('Ready.');
    document.getElementById('search-form').addEventListener('submit', handleQuery);
  } catch (err) {
    console.error(err);
    setStatus('Error: ' + err.message);
  }
}

init();
