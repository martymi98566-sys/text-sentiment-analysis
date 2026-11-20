"use strict";

const $ = (id) => document.getElementById(id);
const examples = {
  mixed: {movie: "好坏混合 · 示例", text: "The acting is convincing. The plot is full of holes. The music is beautiful."},
  opposing: {movie: "同类好坏 · 示例", text: "The lead actor gives a magnificent performance. The supporting actors are wooden and unconvincing."},
  facts: {movie: "剧情叙述 · 示例", text: "A detective arrives in a coastal town. She interviews three witnesses and discovers a letter."},
};
const tones = {mild: "轻微", ordinary: "一般", strong: "强烈"};
const toneOrder = {mild: 0, ordinary: 1, strong: 2};
let names = {}, busy = false, connected = false, pollTimer = null;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function svg(tag, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
}

function connection(ready) {
  connected = ready;
  $("connection").textContent = ready ? "本机服务已连接" : "本机服务未连接";
  $("connection").className = "connection " + (ready ? "ready" : "offline");
  $("analyze").disabled = busy || !ready;
}

function setBusy(value) {
  busy = value;
  for (const node of $("review-form").querySelectorAll("input,textarea,button")) node.disabled = value;
  $("analyze").disabled = value || !connected;
  $("analyze").querySelector("span").textContent = value ? "分析中，请稍候" : "开始分析";
}

function showError(message) {
  $("form-error").textContent = message;
  $("form-error").hidden = !message;
}

function updateCount() {
  const count = Array.from($("review").value).length;
  $("char-count").textContent = `${count.toLocaleString()} / 12,000 字符`;
}

function revealOutput(id) {
  if (window.matchMedia("(max-width: 780px)").matches) {
    $(id).scrollIntoView({behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start"});
  }
}

async function getJSON(url, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(url, {...options, signal: controller.signal});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "请求未完成，请稍后重试。");
    return data;
  } finally { clearTimeout(timeout); }
}

function remember(id) {
  try { sessionStorage.setItem("review-current-job", id); } catch {}
  const url=new URL(window.location.href);url.searchParams.set("job",id);window.history.replaceState(null,"",url);
}
function forget() {
  try { sessionStorage.removeItem("review-current-job"); } catch {}
  const url=new URL(window.location.href);url.searchParams.delete("job");window.history.replaceState(null,"",url);
}
function restoredJob() {
  const linked=new URLSearchParams(window.location.search).get("job");
  if(linked)return linked;
  try { return sessionStorage.getItem("review-current-job"); } catch { return null; }
}

function warning(text) { $("warnings").append(el("p", "", text)); }

function categoryText(unit) {
  return (unit.reason_categories || []).map(c => names[c] || c).join(" / ") || (unit.status === "overall_attitude" ? "总体态度" : "待归类");
}

function selectUnit(index, scrollToCard = true) {
  document.querySelectorAll(".mark.selected,.unit-card.selected").forEach(n => n.classList.remove("selected"));
  const marks = [...$("review-original").querySelectorAll(".mark")].filter(n => n.dataset.units.split(",").includes(String(index)));
  marks.forEach(n => n.classList.add("selected"));
  const card = $("unit-" + index);
  if (card) card.classList.add("selected");
  const target = scrollToCard ? card : marks[0];
  if (target) target.scrollIntoView({behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "nearest"});
}

function renderOriginal(row, units) {
  const target = $("review-original"); target.replaceChildren();
  // Python offsets are Unicode code points; JS string offsets use UTF-16.
  const chars = Array.from(row.text);
  const valid = units.map((u,index) => ({...u,index})).filter(u => Number.isInteger(u.start) && Number.isInteger(u.end) && u.start >= 0 && u.end <= chars.length && u.end > u.start && chars.slice(u.start,u.end).join("") === u.quote);
  const cuts = [...new Set([0, chars.length, ...valid.flatMap(u => [u.start,u.end])])].sort((a,b) => a-b);
  for (let i=0; i<cuts.length-1; i++) {
    const a=cuts[i], b=cuts[i+1], text=chars.slice(a,b).join("");
    const covering = valid.filter(u => u.start <= a && u.end >= b);
    if (!covering.length) { target.append(document.createTextNode(text)); continue; }
    const signs = new Set(covering.map(u => u.polarity));
    const sign = signs.size === 1 ? covering[0].polarity : "mixed";
    const strongest = covering.reduce((a,b) => toneOrder[a.intensity] >= toneOrder[b.intensity] ? a : b);
    const mark = el("span", `mark ${sign} ${strongest.intensity}` + (covering.some(u=>u.context_check_disagrees) ? " conflict" : ""), text);
    mark.tabIndex = 0;
    mark.dataset.units = covering.map(u=>u.index).join(",");
    mark.title = covering.map(u => `${categoryText(u)} · ${u.polarity === "positive" ? "积极" : "消极"} · 语气${tones[u.intensity]}${u.context_check_disagrees ? " · 上下文判断存在分歧" : ""}`).join("\n");
    mark.setAttribute("role", "button"); mark.setAttribute("aria-label", `${text}；${mark.title}`);
    mark.addEventListener("click", () => selectUnit(covering[0].index));
    mark.addEventListener("keydown", e => { if(e.key === "Enter" || e.key === " "){e.preventDefault();selectUnit(covering[0].index);} });
    target.append(mark);
  }
  if (target.textContent !== row.text) throw new Error("原文定位失败，请下载结果核对。");
  if (valid.length !== units.length) warning("部分原文标记未能定位，请下载完整结果核对。");
}

function renderUnits(units) {
  $("unit-list").replaceChildren();
  $("no-units").hidden = !!units.length;
  $("detail-count").textContent = `${units.length} 个片段`;
  units.forEach((u,index) => {
    const card = el("article", "unit-card"); card.id = "unit-" + index;
    card.tabIndex=0; card.setAttribute("role","button"); card.setAttribute("aria-label", "在原文中定位："+u.quote);
    card.append(el("p", "", u.quote));
    const tags=el("div", "unit-tags");
    tags.append(el("span", u.polarity, u.polarity === "positive" ? "积极" : "消极"), el("span", "", categoryText(u)), el("span", "", "语气" + tones[u.intensity]));
    if (u.context_check_disagrees) tags.append(el("span", "caution", "上下文分歧 · 待复核"));
    card.append(tags);
    card.addEventListener("click", () => selectUnit(index,false));
    card.addEventListener("keydown", e => {if(e.key==="Enter" || e.key===" "){e.preventDefault();selectUnit(index,false);}});
    $("unit-list").append(card);
  });
}

function renderComposition(comp) {
  const slices=comp.slices || [], cats=[...new Set(slices.map(s=>s.category))];
  $("type-count").textContent = `${cats.length} 个类型`;
  $("chart-content").hidden = !slices.length;
  $("reason-table-wrap").hidden = !slices.length;
  $("no-reasons").hidden = !!slices.length;
  $("donut").replaceChildren(); $("chart-legend").replaceChildren(); $("reason-table").replaceChildren();
  if (!slices.length) return;
  const chart=svg("svg",{viewBox:"0 0 240 240",role:"img","aria-label":"已归类原因的好评与差评加权构成"});
  const title=svg("title");title.textContent=slices.map(s=>`${names[s.category]}${s.polarity==="positive"?"好评":"差评"}${s.percent.toFixed(1)}%`).join("；");chart.append(title);
  const greens=["#4f9a79","#80b9a0","#326e59","#add0bc","#659b92","#4e7c77"];
  const reds=["#ce7982","#e2a4a7","#b55367","#efc3c3","#c78a9b","#a7657c"];
  let offset=0, pi=0, ni=0;
  for (const s of slices) {
    const color=s.polarity==="positive"?greens[(pi++)%greens.length]:reds[(ni++)%reds.length];
    const circle=svg("circle",{cx:120,cy:120,r:88,fill:"none",stroke:color,"stroke-width":31,pathLength:100,"stroke-dasharray":`${s.percent} ${100-s.percent}`,"stroke-dashoffset":-offset,transform:"rotate(-90 120 120)"});
    const tooltip=svg("title");tooltip.textContent=`${names[s.category]} · ${s.polarity==="positive"?"好评":"差评"} ${s.percent.toFixed(1)}%`;circle.append(tooltip);chart.append(circle);offset+=s.percent;
    const legend=el("div","legend-row"),dot=svg("svg",{viewBox:"0 0 10 10","aria-hidden":"true"});dot.append(svg("rect",{width:10,height:10,rx:2,fill:color}));
    legend.append(dot,el("span","legend-name",`${names[s.category] || s.category} · ${s.polarity==="positive"?"好评":"差评"}`),el("strong","",s.percent.toFixed(1)+"%"));
    $("chart-legend").append(legend);
  }
  const center=svg("text",{x:120,y:119,"text-anchor":"middle",class:"donut-number"}); center.textContent=String(cats.length);
  const sub=svg("text",{x:120,y:141,"text-anchor":"middle",class:"donut-label"});sub.textContent="已识别原因类型";
  chart.append(center,sub);$("donut").append(chart);
  for(const c of cats){
    const pos=slices.filter(s=>s.category===c&&s.polarity==="positive").reduce((a,s)=>a+s.percent,0);
    const neg=slices.filter(s=>s.category===c&&s.polarity==="negative").reduce((a,s)=>a+s.percent,0);
    const row=el("tr");row.append(el("td","",names[c]||c),el("td",pos?"pos":"none",pos?pos.toFixed(1)+"%":"—"),el("td",neg?"neg":"none",neg?neg.toFixed(1)+"%":"—"),el("td","",(pos+neg).toFixed(1)+"%"));$("reason-table").append(row);
  }
}

function renderResult(data,id) {
  const row=data.result;names=Object.fromEntries(data.categories.map(c=>[c.id,c.name]));
  const units=row.emotion_units || [], comp=row.composition;
  $("result-movie").textContent=row.movie_title;
  $("overall").textContent=row.overall.overall_sentiment;
  $("overall").className=row.overall.overall_sentiment === "积极" ? "positive" : "negative";
  $("unit-count").textContent=String(units.length);
  const unknown=units.filter(u=>!(u.reason_categories || []).length && u.status!=="overall_attitude").length;
  $("pending-count").textContent=unknown?`${unknown} 段原因待归类`:"含总体态度与具体评价";
  $("coverage").textContent=(100*comp.evidence_word_coverage).toFixed(1)+"%";
  $("warnings").replaceChildren();
  renderOriginal(row,units);renderUnits(units);renderComposition(comp);
  if(unknown)warning(`${unknown} 个情绪片段的具体原因尚未归类，不计入下方占比。因此饼图不代表完整评论的全部原因。`);
  const conflicts=units.filter(u=>u.context_check_disagrees).length;
  if(conflicts)warning(`${conflicts} 个片段的局部判断与上下文复核存在分歧，已用虚线标记，请结合原文核对。`);
  if(row.overall.overall_transformer_truncated)warning("评论较长，整体情绪判断中的部分模型只使用了前段文字。");
  if(!units.length)warning("整体情绪仍按积极／消极二选一；如果原文只有客观叙述，请勿将该判断视为真实情绪。");
  $("download-html").href=`/api/jobs/${id}/report`;$("download-json").href=`/api/jobs/${id}/json`;
  $("progress").hidden=true;$("empty-state").hidden=true;$("result").hidden=false;
  revealOutput("result");
}

async function poll(id, failures=0) {
  clearTimeout(pollTimer);
  try {
    const job=await getJSON(`/api/jobs/${id}`);
    connection(true);
    if(job.status==="completed"){
      const data=await getJSON(`/api/jobs/${id}/result`);renderResult(data,id);setBusy(false);return;
    }
    if(job.status==="failed")throw new Error(job.error);
    $("progress-title").textContent=job.status==="queued"?"正在等待分析":"正在分析评论";
    $("progress-detail").textContent=job.status==="queued"?`当前排队位置：${job.queue_position}。前面的任务结束后会自动开始。`:"正在识别情绪片段与具体原因，无需人工操作。";
    $("elapsed").textContent=`${job.status==="queued"?"已等待":"已用时"} ${job.elapsed_seconds} 秒`;
    pollTimer=setTimeout(()=>poll(id),1500);
  } catch(error) {
    if(failures<3 && !String(error.message).includes("没有找到") && !String(error.message).includes("本次分析未完成")){
      pollTimer=setTimeout(()=>poll(id,failures+1),2000);return;
    }
    showError(error.message || "服务连接中断，请检查服务后刷新页面。");
    $("progress").hidden=true;$("empty-state").hidden=false;setBusy(false);
    if(String(error.message).includes("本次分析未完成") || String(error.message).includes("没有找到"))forget();
  }
}

$("review").addEventListener("input",updateCount);
$("clear").addEventListener("click",()=>{$("review").value="";$("text-file").value="";updateCount();showError("");$("review").focus();});
document.querySelectorAll("[data-example]").forEach(button=>button.addEventListener("click",()=>{
  const example=examples[button.dataset.example];$("movie").value=example.movie;$("review").value=example.text;showError("");updateCount();$("review").focus();
}));
$("text-file").addEventListener("change",async event=>{
  const file=event.target.files[0];if(!file)return;
  try{
    if(file.size>65536)throw new Error("请选择64KB以内的UTF-8文本文件。");
    const text=new TextDecoder("utf-8",{fatal:true}).decode(await file.arrayBuffer());
    if(Array.from(text).length>12000)throw new Error("文件内容超过12000字符，请保留一条较短评论。");
    $("review").value=text;updateCount();showError("");
  }catch(error){showError(error.message.includes("encoded")?"文件不是UTF-8编码，请转换为UTF-8后重试。":error.message);}
  event.target.value="";
});
$("review-form").addEventListener("submit",async event=>{
  event.preventDefault();if(busy)return;showError("");
  const text=$("review").value;
  if(!text.trim())return showError("请先输入一条评论。");
  if(Array.from(text).length>12000)return showError("评论超过12000字符，请缩短后重试。");
  if(!/[A-Za-z]/.test(text))return showError("当前版本支持英文评论，请输入英文原文。");
  setBusy(true);$("result").hidden=true;$("empty-state").hidden=true;$("progress").hidden=false;
  $("progress-title").textContent="正在提交评论";$("progress-detail").textContent="准备本次分析。";$("elapsed").textContent="已用时 0 秒";
  revealOutput("progress");
  try{
    const job=await getJSON("/api/jobs",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({movie:$("movie").value,text})});
    remember(job.id);poll(job.id);
  }catch(error){showError(error.message || "无法连接本机服务。");$("progress").hidden=true;$("empty-state").hidden=false;setBusy(false);}
});

(async()=>{
  updateCount();
  try{
    const health=await getJSON("/api/health");names=Object.fromEntries(health.categories.map(c=>[c.id,c.name]));connection(true);
    const id=restoredJob();
    if(id && /^[a-f0-9]{32}$/.test(id)){
      try{
        const job=await getJSON(`/api/jobs/${id}`);$("review").value=job.request.text;$("movie").value=job.request.movie==="未提供片名"?"":job.request.movie;updateCount();
        setBusy(true);$("empty-state").hidden=true;$("progress").hidden=false;poll(id);
      }catch{forget();}
    }
  }catch{connection(false);showError("本机分析服务未连接，请启动服务后刷新页面。");}
})();
