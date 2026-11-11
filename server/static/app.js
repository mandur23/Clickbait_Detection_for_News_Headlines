"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const form = $("predict-form");
  const titleInput = $("title");
  const bodyInput = $("body");
  const submitButton = $("submit");
  const errorBox = $("error");
  const resultBox = $("result");

  const info = {
    model: "",
    is_mock: false,
    accepts_body: false,
    max_title_chars: 200,
    max_body_chars: 20000,
    history_enabled: false,
  };

  // 확신이 낮을수록 말을 누그러뜨린다. 위에서부터 처음 맞는 줄을 쓴다.
  const VERDICTS = {
    clickbait: [
      [0.8, "낚시성 제목일 가능성이 높습니다"],
      [0.6, "낚시성 제목으로 보입니다"],
      [0.5, "낚시성 쪽에 조금 더 가깝지만, 판단이 애매합니다"],
    ],
    normal: [
      [0.8, "일반 제목일 가능성이 높습니다"],
      [0.6, "일반 제목으로 보입니다"],
      [0.5, "일반 쪽에 조금 더 가깝지만, 판단이 애매합니다"],
    ],
  };
  const LABEL_KO = { clickbait: "낚시성", normal: "일반" };

  // 서버의 len()과 같게 코드 포인트 단위로 센다(이모지 하나 = 1자).
  const charCount = (text) => [...text].length;

  function verdictFor(label, prob) {
    const row = VERDICTS[label].find(([threshold]) => prob >= threshold);
    return row ? row[1] : VERDICTS[label].at(-1)[1];
  }

  function percent(p) {
    return `${(p * 100).toFixed(1)}%`;
  }

  function updateCounter(input, counter, max) {
    const n = charCount(input.value);
    counter.textContent = `${n.toLocaleString("ko-KR")} / ${max.toLocaleString("ko-KR")}`;
    counter.classList.toggle("over", n > max);
  }

  const updateTitleCounter = () => updateCounter(titleInput, $("title-counter"), info.max_title_chars);
  const updateBodyCounter = () => updateCounter(bodyInput, $("body-counter"), info.max_body_chars);

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function clearError() {
    errorBox.textContent = "";
    errorBox.hidden = true;
  }

  function clientCheck(title, body) {
    if (title.trim() === "") return "제목을 입력해 주세요.";
    const n = charCount(title);
    if (n > info.max_title_chars) {
      return `제목은 ${info.max_title_chars}자 이하로 입력해 주세요. (현재 ${n}자)`;
    }
    if (body !== null && charCount(body) > info.max_body_chars) {
      const max = info.max_body_chars.toLocaleString("en-US");
      const len = charCount(body).toLocaleString("en-US");
      return `본문은 ${max}자 이하로 입력해 주세요. (현재 ${len}자)`;
    }
    return null;
  }

  function setBar(kind, prob) {
    $(`bar-${kind}`).style.width = `${(prob * 100).toFixed(1)}%`;
    $(`pct-${kind}`).textContent = percent(prob);
  }

  function showResult(title, result) {
    const prob = result.label === "clickbait" ? result.prob_clickbait : result.prob_normal;
    resultBox.dataset.label = result.label;
    $("verdict").textContent = verdictFor(result.label, prob);
    $("verdict-title").textContent = title;
    $("stamp-label").textContent = LABEL_KO[result.label];
    $("stamp-sub").textContent = `확률 ${percent(prob)}`;
    // 반반에 가까우면 도장을 점선으로 흐리게 찍어 확신이 낮다는 것을 보인다.
    $("stamp").dataset.weak = String(prob < 0.6);
    $("result-empty").hidden = true;

    for (const kind of ["clickbait", "normal"]) {
      $(`bar-${kind}`).style.width = "0";
      $(`pct-${kind}`).textContent = "0%";
    }
    resultBox.hidden = false;
    resultBox.classList.remove("stamped");
    // 너비 0과 도장 없는 상태를 먼저 그리게 해야 막대와 도장 움직임이 매번 처음부터 다시 일어난다.
    void resultBox.offsetWidth;
    resultBox.classList.add("stamped");
    setBar("clickbait", result.prob_clickbait);
    setBar("normal", result.prob_normal);
  }

  function setLoading(loading) {
    submitButton.disabled = loading;
    submitButton.textContent = loading ? "판별 중…" : "판별하기";
    form.setAttribute("aria-busy", String(loading));
  }

  async function submit() {
    const title = titleInput.value;
    const body = info.accepts_body && bodyInput.value.trim() !== "" ? bodyInput.value : null;
    const problem = clientCheck(title, body);
    if (problem) {
      showError(problem);
      titleInput.focus();
      return;
    }

    clearError();
    setLoading(true);
    try {
      const payload = body === null ? { title } : { title, body };
      let response;
      try {
        response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      } catch {
        showError("서버에 연결할 수 없습니다. 서버가 켜져 있는지 확인한 뒤 다시 시도해 주세요.");
        return;
      }
      const data = await response.json().catch(() => null);
      if (!response.ok || data === null) {
        showError(data?.error?.message ?? "판별 중 문제가 생겼습니다. 잠시 후 다시 시도해 주세요.");
        return;
      }
      showResult(title.trim(), data);
      loadHistory();
    } finally {
      setLoading(false);
    }
  }

  function formatTime(iso) {
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "";
    return date.toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" });
  }

  function renderHistory(records) {
    const list = $("history-list");
    list.replaceChildren();
    $("history-empty").hidden = records.length > 0;
    for (const record of records) {
      const item = document.createElement("li");

      const tag = document.createElement("span");
      tag.className = "tag";
      tag.dataset.label = record.label;
      tag.textContent = LABEL_KO[record.label] ?? record.label;

      const title = document.createElement("span");
      title.textContent = record.title;

      const meta = document.createElement("span");
      meta.className = "history-meta";
      meta.textContent = `낚시성 ${percent(record.prob_clickbait)} · ${formatTime(record.created_at)}`;

      item.append(tag, title, meta);
      list.append(item);
    }
  }

  async function loadHistory() {
    if (!info.history_enabled) return;
    try {
      const response = await fetch("/api/history?limit=10");
      if (response.ok) renderHistory(await response.json());
    } catch {
      // 기록은 부가 기능이라 실패해도 판별 화면은 그대로 둔다.
    }
  }

  async function loadInfo() {
    try {
      const response = await fetch("/api/info");
      Object.assign(info, await response.json());
    } catch {
      $("model-name").textContent = "모델 정보를 불러오지 못했습니다.";
      return;
    }
    $("demo-notice").hidden = !info.is_mock;
    $("body-field").hidden = !info.accepts_body;
    $("history").hidden = !info.history_enabled;
    $("model-name").textContent = info.is_mock
      ? "모델: 데모용 가짜 판별기 (학습된 모델 없음)"
      : `모델: ${info.model}`;
    updateTitleCounter();
    if (info.accepts_body) updateBodyCounter();
    loadHistory();
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!submitButton.disabled) submit();
  });

  titleInput.addEventListener("keydown", (event) => {
    // 한글 조합 중 Enter는 글자를 확정하는 키라서 제출로 보지 않는다.
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  titleInput.addEventListener("input", updateTitleCounter);
  bodyInput.addEventListener("input", updateBodyCounter);

  for (const chip of document.querySelectorAll(".chip")) {
    chip.addEventListener("click", () => {
      titleInput.value = chip.textContent;
      updateTitleCounter();
      titleInput.focus();
      form.requestSubmit();
    });
  }

  $("today").textContent = new Date().toLocaleDateString("ko-KR", {
    year: "numeric",
    month: "long",
    day: "numeric",
    weekday: "long",
  });
  updateTitleCounter();
  loadInfo();
})();
