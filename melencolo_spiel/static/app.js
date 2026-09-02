const processButton = document.getElementById("processBtn");
const startButton = document.getElementById("startBtn");
const textButton = document.getElementById("textBtn");
const txt_area = document.getElementById("inputText");
const table = document.getElementById("resultTable");
//const tbody = table.querySelector("tbody");



processButton.onclick = async () => {
    const text = document.getElementById("inputText").value;

    // try {
    const res = await fetch("http://127.0.0.1:8000/align", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
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

    txt_area.style.display = "none";
    table.style.display = "table";
    btnRun.style.display = "none";
    btnSave.style.display = "block";

    console.log("before: "+window.innerWidth);
    console.log("before: "+document.documentElement.clientWidth);

};
startButton.onclick = async () => {

    const selected = [];

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

    console.log("after: " + window.innerWidth);
    console.log("after: " + document.documentElement.clientWidth);
};
textButton.onclick = async () => {

}