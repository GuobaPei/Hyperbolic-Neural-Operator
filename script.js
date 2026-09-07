const benchmarks = {
  elasticity: {
    name: "Elasticity",
    geometry: "Point Cloud",
    hno: "0.0037",
    second: "0.0064",
    gain: "42.2%"
  },
  navier: {
    name: "Navier-Stokes",
    geometry: "Regular Grid",
    hno: "0.0676",
    second: "0.0892",
    gain: "24.2%"
  },
  darcy: {
    name: "Darcy",
    geometry: "Regular Grid",
    hno: "0.0045",
    second: "0.0054",
    gain: "16.7%"
  },
  plasticity: {
    name: "Plasticity",
    geometry: "Structured Mesh",
    hno: "0.0009",
    second: "0.0012",
    gain: "25.0%"
  },
  airfoil: {
    name: "Airfoil",
    geometry: "Structured Mesh",
    hno: "0.0048",
    second: "0.0053",
    gain: "9.4%"
  },
  pipe: {
    name: "Pipe",
    geometry: "Structured Mesh",
    hno: "0.0027",
    second: "0.0042",
    gain: "35.7%"
  }
};

const pickerButtons = document.querySelectorAll(".benchmark-picker button");
const resultName = document.getElementById("result-name");
const resultGeometry = document.getElementById("result-geometry");
const resultHno = document.getElementById("result-hno");
const resultSecond = document.getElementById("result-second");
const resultGain = document.getElementById("result-gain");

function renderBenchmark(key) {
  const item = benchmarks[key];
  if (!item) return;

  resultName.textContent = item.name;
  resultGeometry.textContent = item.geometry;
  resultHno.textContent = item.hno;
  resultSecond.textContent = item.second;
  resultGain.textContent = item.gain;

  pickerButtons.forEach((button) => {
    const active = button.dataset.key === key;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
}

pickerButtons.forEach((button) => {
  button.addEventListener("click", () => renderBenchmark(button.dataset.key));
});

const copyBibButton = document.getElementById("copy-bib");
const bibtex = document.getElementById("bibtex");

copyBibButton.addEventListener("click", async () => {
  const text = bibtex.textContent;

  try {
    await navigator.clipboard.writeText(text);
    copyBibButton.textContent = "Copied";
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(bibtex);
    selection.removeAllRanges();
    selection.addRange(range);
    copyBibButton.textContent = "Selected";
  }

  window.setTimeout(() => {
    copyBibButton.textContent = "Copy BibTeX";
  }, 1400);
});
