let translationOutput = {
    "has_content": false,
    "weight": null,
    "input_tokens": null,
    "target_tokens": null
}

class Token {
    constructor(index, token) {
        this.index = index;
        this.token = token;
        this.linkedLines = new Array();
    }

    addLine(line) {
        this.linkedLines.push(line);
    }

    setElement(element) {
        this.element = element;
        element.objectRef = this;
    }

    render() {
        let tokenDiv = document.createElement("div");
        tokenDiv.innerText = this.token;
        return tokenDiv;
    }
}

class Line {
    constructor(leftYOffset, rightYOffset, weight, containerSize) {
        this.leftYOffset = leftYOffset;
        this.rightYOffset = rightYOffset;
        this.weight = weight;
        [this.height, this.width] = containerSize;
    }

    setEndpoints(inputToken, targetToken) {
        this.inputToken = inputToken;
        this.targetToken = targetToken;
    }

    setElement(element) {
        this.element = element;
        element.objectRef = this;
    }

    render() {
        return `<a onclick="alert('w: ${this.weight}!');"><line class="line" x1="0" y1="${this.leftYOffset}" x2="${this.width}" y2="${this.rightYOffset}" /></a>`;
    }
}

function loadTokens(table, num_tokens, tokens) {
    for (let i = 0; i < num_tokens; i++) {
        let token = new Token(i, tokens[i]);
        let newRow = table.insertRow(-1);
        let cell = newRow.insertCell(0);
        let tokenDiv = token.render();
        newRow.classList.add("token-cell");
        cell.appendChild(tokenDiv);
        cell.setAttribute("onclick", "displayTokenAttentionLine(this)");
        token.setElement(cell);
    }
}

function plotLines(inputTable, targetTable, inputLength, targetLength, weightMatrix) {
    let inputTableRect = inputTable.getBoundingClientRect();
    let targetTableRect = targetTable.getBoundingClientRect();
    let containerRect = document.getElementById("line-plot").getBoundingClientRect();

    let plotHeight = Math.max(inputTableRect.height, targetTableRect.height);
    let containerSize = [plotHeight, containerRect.width];

    let a, inputRows, targetRows;
    [a, ...inputRows] = inputTable.rows;
    [a, ...targetRows] = targetTable.rows;

    let lineStrings = "";
    let lineObjs = [];
    for (let i = 0; i < inputLength; i++) {
        let inputTokenObj = inputRows[i].children[0].objectRef;
        let inputTokenBounding = inputRows[i].getBoundingClientRect();
        inputOffsetY = (inputTokenBounding.bottom + inputTokenBounding.top) / 2;

        for (let j = 0; j < targetLength; j++) {
            let targetTokenObj = targetRows[j].children[0].objectRef;
            let targetTokenBounding = targetRows[j].getBoundingClientRect();
            targetOffsetY = (targetTokenBounding.bottom + targetTokenBounding.top) / 2;

            let line = new Line(inputOffsetY - inputTableRect.top, targetOffsetY - targetTableRect.top, weightMatrix[j][i], containerSize);

            inputTokenObj.addLine(line);
            targetTokenObj.addLine(line);
            line.setEndpoints(inputTokenObj, targetTokenObj);

            let svg = line.render();
            lineStrings += svg;
            lineObjs.push(line);
        }
    }

    let svgString = `<svg id="line-svg" height="${containerSize[0]}" width="${containerSize[1]}">${lineStrings}</svg>`;

    document.getElementById("line-plot").innerHTML = svgString;

    let lines = document.getElementById("line-svg").children;

    for (let i = 0; i < lines.length; i++) {
        let line = lines[i].children[0];
        let lineObj = lineObjs[i];
        lineObj.setElement(line);
        line.style.opacity = lineObj.weight;
    }
}

function displayTokenAttentionLine(element) {
    let lines = document.getElementById("line-svg").children;

    let cells = document.getElementsByClassName("token-cell");
    for(let cell of cells) {
        cell.children[0].style.backgroundColor = "transparent";
    }

    element.style.backgroundColor = "rgba(98, 157, 240, 0.6)";

    for (let line of lines) {
        line.style.visibility='hidden';
    }

    for (let lineObj of element.objectRef.linkedLines) {
        lineObj.element.parentElement.style.visibility='visible';
    }
}

function visualizeLine() {
    let attentionMapIndex = document.getElementById("number-input").value;
    let inputTokenTable = document.getElementById("input-token");
    let targetTokenTable = document.getElementById("target-token");

    let inputLength = translationOutput["input_tokens"].length;
    let targetLength = translationOutput["target_tokens"].length;

    plotLines(inputTokenTable, targetTokenTable, inputLength, targetLength, translationOutput["weight"][attentionMapIndex]);
}

function alignLanguage() {
    let languageContent = document.getElementById("input-language");
    let languages = document.getElementsByClassName("language-name");

    Array.prototype.forEach.call(languages, language => {
        language.style.width = languageContent.scrollWidth + 'px';
    });

}

function setAutoResize() {
    let input = document.getElementById("input-language");
    let target = document.getElementById("target-language");

    input.addEventListener('input', () => {
        autoResize(input, target);
    }, false)
}

function autoResize(input, target) {
    input.style.height = 'auto';
    target.style.height = "auto";
    let height = Math.max(input.scrollHeight, target.scrollHeight);
    input.style.height = height  - 20 + 'px';
    target.style.height = height  - 20 + 'px';
}

function swapLanguage() {
    let input = document.getElementById("input-language-name");
    let target = document.getElementById("target-language-name");

    let temp = target.innerText;
    target.innerText = input.innerText;
    input.innerText = temp;
}

function postData() {
    let inputText = document.getElementById("input-language").value;
    let inputLanguage = document.getElementById("input-language-name");
    let targetLanguage = document.getElementById("target-language-name");

    let targetTextArea = document.getElementById("target-language");

    let data = {
        input_language: inputLanguage.innerText.toLowerCase(),
        target_language: targetLanguage.innerText.toLowerCase(),
        text: inputText
    }

    targetTextArea.value = "...";

    clearTable();

    fetch('/translate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        targetTextArea.value = data["translation"];
        translationOutput["has_content"] = true;
        translationOutput["weight"] = data["weight"];
        translationOutput["input_tokens"] = data["input_tokens"];
        translationOutput["target_tokens"] = data["target_tokens"];
        plotAttention();
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function clearData() {
    let textCells = document.getElementsByClassName("autoresizing-textarea");
    Array.prototype.forEach.call(textCells, cell => {
        cell.value = "";
        cell.style.height = 'auto';
    });
    clearTable()
}

function clearTable() {
    document.getElementById("line-plot").innerHTML = "";
    let tables = document.querySelectorAll("table");
    for(let table of tables) {
        while(table.rows.length > 1) {
            table.deleteRow(-1);
        }
    }
}

function showToast(info) {
    var toast = document.getElementById("message");
    toast.innerText = info;
    toast.style.transform = `translateX(-${toast.offsetWidth/2}px)`;

    toast.classList.add("show");
    setTimeout(function(){ toast.classList.remove("show"); }, 3000);
}

function shutDown(element) {
    element.classList.add("block-btn");
    element.style.backgroundColor = 'yellow';
    element.style.width = '15px';
    element.style.height = '15px';
    element.title = "Shutting down";

    const spinElement = document.createElement('div');
    spinElement.classList.add('change-state');
    element.appendChild(spinElement);

    let  data = "shutdown"

    let status = ""

    fetch('', {
        method: 'POST',
        headers: {
        'Content-Type': 'application/json'
        },
        body: data
    }).then(response => response.text())
        .then(data => {
            tar_text = JSON.parse(data).status
            if (tar_text === "shutdown") {
                status = "Server disconnected!";
            }
            else {
                throw new Error("Server disconnected!");
            }
        })
        .catch(function(error) {
                status = "Errors occured when shutting down server!";
    })
    .finally(() => {
            element.style.borderRadius = '0';
            element.style.backgroundColor = 'red';
            const spinElement = element.querySelector('.change-state');
            element.removeChild(spinElement);
            element.title = "Disconnected";
            currentState = 'square';
            showToast(status);
    });
}

function render() {
    alignLanguage();
    setAutoResize();
    document.getElementById("swap-btn").setAttribute("onclick", "swapLanguage();")
}

function plotAttention() {
    let numberInput = document.getElementById("number-input");
    numberInput.value = 0;
    let attentionMapIndex = numberInput.value;

    let inputTokenTable = document.getElementById("input-token");
    let targetTokenTable = document.getElementById("target-token");

    let inputLength = translationOutput["input_tokens"].length;
    let targetLength = translationOutput["target_tokens"].length;

    loadTokens(inputTokenTable, inputLength, translationOutput["input_tokens"]);
    loadTokens(targetTokenTable, targetLength, translationOutput["target_tokens"]);

    plotLines(inputTokenTable, targetTokenTable, inputLength, targetLength, translationOutput["weight"][attentionMapIndex]);
}