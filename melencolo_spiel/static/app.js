/*RESPONSIVE MELENCOLO*/
const processButton = document.getElementById("processBtn");
const startButton = document.getElementById("startBtn");

const divButtons = document.getElementById("buttonContainer");
const divBtnProcess = document.getElementById("btnContainer_process");
const divBtnStart = document.getElementById("btnContainer_start");

const btnClear = document.getElementById("clearButton");
const btnSave = document.getElementById("saveButton");

const textButton = document.getElementById("textBtn");
const translateBtn = document.getElementById("translateBtn");
const tableBtn = document.getElementById("tableBtn");

const txt_area = document.getElementById("pseudoCanvas");
const table = document.getElementById("table");

const hourGlass = document.getElementById("hourglass");
// Store only the icon markup, without duplicating the ID
const hourGlassHTML = hourGlass.innerHTML.trim();
// Hide the original HTML element
hourGlass.style.display = "none";

let input_txt = "";
let translation = "";


/* ================================================================
   PROCESS TEXT
   ================================================================ */

processButton.onclick = async () => {
    // Reuse the icon in buttons
    processButton.innerHTML =
        `<span class="material-symbols-outlined">${hourGlassHTML}</span>`;
    processButton.disabled = true;

    input_txt = txt_area.value;

    /*
        Clear previous table contents before adding the new result.
    */
    const thead = table.querySelector("thead");
    const tbody = table.querySelector("tbody");

    thead.innerHTML = "";
    tbody.innerHTML = "";

    const head_row = document.createElement("tr");

    head_row.innerHTML = `
        <th>keep</th>
        <th>German</th>
        <th>English</th>
`;

    thead.appendChild(head_row);


    try {

        /*
            IMPORTANT:

            Do NOT use 127.0.0.1 here.

            A phone accessing the application sees 127.0.0.1
            as the phone itself.

            A relative URL means:
                "use the same server that delivered this page".
        */
        const res = await fetch("/align", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                text: input_txt
            })
        });


        if (!res.ok) {
            throw new Error(`Server returned HTTP ${ res.status } `);
        }


        const data = await res.json();

        translation = data.translation;

        console.log(data);


        data.pairs.forEach((pair, index) => {

            const row = document.createElement("tr");

            row.innerHTML = `
                <td>
                    <input
                        type="checkbox"
                        data-index="${index}"
                        aria-label="Keep ${pair.de}"
                    >
                </td>

                <td>${pair.de}</td>

                <td>${pair.en}</td>
`;

            tbody.appendChild(row);
        });


        hideItems();

        table.classList.remove("hide");
        table.classList.add("show-grid");

        divBtnStart.classList.remove("hide");
        divBtnStart.classList.add("show-grid");

        btnClear.classList.remove("hide");
        btnClear.classList.add("show-btn");

        btnSave.classList.remove("hide");
        btnSave.classList.add("show-btn");

    } catch (error) {

        console.error("Processing failed:", error);

        alert(
            "Unable to process the text.\n\n" +
            "Please check that the backend server is running."
        );

    } finally {

        processButton.innerHTML = "▶";
        processButton.disabled = false;
    }
};


/* ================================================================
   START GAME
   ================================================================ */

startButton.onclick = async () => {

    const selected = [];

    // Reuse the icon in buttons
    startButton.innerHTML =
        `<span class="material-symbols-outlined">${hourGlassHTML}</span>`;
    startButton.disabled = true;

    const tbody = table.querySelector("tbody");

    /*
        Only process rows that actually contain a checkbox.

        This is safer than assuming that every <tr> has exactly
        three cells.
    */
    const rows = tbody.querySelectorAll("tr");

    rows.forEach((row) => {

        const checkbox = row.querySelector(
            'input[type="checkbox"][data-index]'
        );

        /*
            Ignore incomplete/non-data rows.
        */
        if (!checkbox) {
            return;
        }

        /*
            Get the actual data cells directly.
        */
        const cells = row.querySelectorAll(":scope > td");

        console.log("VALORE CELLE: "+cells)

        /*
            A valid word-pair row must contain:
                0 = checkbox
                1 = German
                2 = English
        */
        if (cells.length < 3) {
            console.warn(
                "Ignoring incomplete table row:",
                row
            );
            return;
        }

        if (checkbox.checked) {

            selected.push({
                de: cells[1].textContent.trim(),
                en: cells[2].textContent.trim()
            });
        }
    });


    console.log("Selected pairs:", selected);


    /*
        Optional but useful:
        don't start the game if nothing was selected.
    */
    if (selected.length === 0) {

        alert(
            "Please select at least one word pair before starting the game."
        );

        startButton.innerHTML = "▶";
        startButton.disabled = false;

        return;
    }


    try {

        const res = await fetch("/save_pairs", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                pairs: selected
            })
        });


        if (!res.ok) {
            throw new Error(
                `Server returned HTTP ${ res.status } `
            );
        }


        const data = await res.json();


        if (data.status !== "ok") {
            throw new Error(
                "Backend did not confirm saved pairs."
            );
        }


        /*
            Start Phaser game.
        */
        startGame();


        /*
            Restore the text controls.
        */
        hideItems();

        txt_area.value = input_txt;

        txt_area.classList.remove("hide");
        txt_area.classList.add("show-grid");

        divButtons.classList.remove("hide");
        divButtons.classList.add("show-grid");


        btnClear.classList.remove("hide");
        btnClear.classList.add("show-btn");

        btnSave.classList.remove("hide");
        btnSave.classList.add("show-btn");


    } catch (error) {

        console.error(
            "Unable to start game:",
            error
        );

        alert(
            "Unable to start the game.\n\n" +
            "Please check that the backend server is running."
        );

    } finally {

        startButton.innerHTML = "▶";
        startButton.disabled = false;
    }
};

/* ================================================================
   SAVE GAME
   ================================================================ */

btnSave.onclick = async () => {

};

/* ================================================================
   CLEAR BUTTON
   ================================================================ */

btnClear.onclick = () => {

    txt_area.value = "";

    input_txt = "";
    translation = "";

    table.querySelector("thead").innerHTML = "";
    table.querySelector("tbody").innerHTML = "";

    hideItems();

    txt_area.classList.remove("hide");
    txt_area.classList.add("show-grid");

    divBtnProcess.classList.remove("hide");
    divBtnProcess.classList.add("show-grid");


    btnClear.classList.remove("hide");
    btnClear.classList.add("show-btn");

    txt_area.focus();

    resetGame();

};


/* ================================================================
   ORIGINAL TEXT BUTTON
   ================================================================ */

textButton.onclick = () => {

    hideItems();

    txt_area.value = input_txt;

    txt_area.classList.remove("hide");
    txt_area.classList.add("show-grid");

    divButtons.classList.remove("hide");
    divButtons.classList.add("show-grid");

    btnClear.classList.remove("hide");
    btnClear.classList.add("show-btn");

    btnSave.classList.remove("hide");
    btnSave.classList.add("show-btn");
};


/* ================================================================
   TRANSLATION BUTTON
   ================================================================ */

translateBtn.onclick = () => {

    hideItems();

    txt_area.value = translation;

    txt_area.classList.remove("hide");
    txt_area.classList.add("show-grid");

    divButtons.classList.remove("hide");
    divButtons.classList.add("show-grid");

    btnClear.classList.remove("hide");
    btnClear.classList.add("show-btn");

    btnSave.classList.remove("hide");
    btnSave.classList.add("show-btn");
};


/* ================================================================
   TABLE BUTTON
   ================================================================ */

tableBtn.onclick = () => {

    hideItems();

    table.classList.remove("hide");
    table.classList.add("show-grid");

    divButtons.classList.remove("hide");
    divButtons.classList.add("show-grid");

    btnClear.classList.remove("hide");
    btnClear.classList.add("show-btn");

    btnSave.classList.remove("hide");
    btnSave.classList.add("show-btn");
};


/* ================================================================
   INITIAL STATE
   ================================================================ */

document.addEventListener("DOMContentLoaded", () => {

    hideItems();

    txt_area.classList.remove("hide");
    txt_area.classList.add("show-grid");

    divBtnProcess.classList.remove("hide");
    divBtnProcess.classList.add("show-grid");

    btnClear.classList.remove("hide");
    btnClear.classList.add("show-btn");

});


/* ================================================================
   HIDE ALL SECONDARY UI
   ================================================================ */

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

    //startButton.classList.remove("show-btn");
    //startButton.classList.add("hide");

    btnClear.classList.remove("show-btn");
    btnClear.classList.add("hide");

    btnSave.classList.remove("show-btn");
    btnSave.classList.add("hide");
}
