const processButton = document.getElementById("processBtn");
const startButton = document.getElementById("startBtn");
const divButtons = document.getElementById("buttonContainer");
const divBtnProcess = document.getElementById("btnContainer_process");
const divBtnStart = document.getElementById("btnContainer_start");
const btnClear = document.getElementById("clearButton");
const textButton = document.getElementById("textBtn");
const txt_area = document.getElementById("pseudoCanvas");
const table = document.getElementById("table");
const input_txt;
const translation;

processButton.onclick = async () => {
    processButton.innerHTML = "Loading...";
    tbody = table.querySelector("tbody");
    input_txt = txt_area.value;

    // try {
    const res = await fetch("http://127.0.0.1:8000/align", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input_txt })
    });

    const data = await res.json();

    tbody.innerHTML = "";

    data.pairs.forEach((pair, index) => {
        const row = document.createElement("tr");

        row.innerHTML = `
                <td>
                    <input
                    type="checkbox"
                    unchecked
                    data-index="${index}">
                </td>
                <td>${pair.de}</td>
                <td>${pair.en}</td>
            `;

        tbody.appendChild(row);
    });

    hideItems();
    processButton.innerHTML = '▶';
    table.classList.add("show-grid");
    divBtnStart.classList.add("show-grid");
    
};
startButton.onclick = async () => {

    const selected = [];
    startButton.innerHTML = "Loading..."
    tbody.querySelectorAll("tr").forEach((row, i) => {

        const checked = row.querySelector("input").checked;

        if (checked) {

            selected.push({
                de: row.cells[1].textContent,
                en: row.cells[2].textContent
            });

        }

    });

    console.log(selected);

    // POST to FastAPI if desired

    const res = await fetch("/save_pairs", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            pairs: selected
        })
    });

    const data = await res.json()
    if (data.status == "ok")
        startGame();

    hideItems();
    startButton.innerHTML = '▶';
    txt_area.value = input_txt;
    txt_area.classList.add("show-grid");
    divButtons.classList.add("show-grid");

};
btnClear.onclick = () => {

};
textButton.onclick = async () => {

}
document.addEventListener("DOMContentLoaded", function () {
    divBtnProcess.classList.remove("hide");
    divBtnProcess.classList.add("show-grid");
    divBtnStart.classList.remove("show-grid");
    divBtnStart.classList.add("hide");
    divButtons.classList.remove("show-grid");
    divButtons.classList.add("hide");
})

function hideItems() {
    txt_area.classList.remove("show-grid");
    txt_area.classList.add("hide");
    table.classList.remove("show-grid");
    table.classList.add("hide");
    divBtnProcess.classList.remove("show-grid");
    divBtnProcess.classList.add("hide");
    divBtnStart.classList.remove("show-grid");
    divBtnStart.classList.add("hide");
    divButtons.classList.remove("show-grid");
    divButtons.classList.add("hide");
}