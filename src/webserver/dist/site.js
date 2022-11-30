

// the function to be called when the generate name button is pressed
function generateName() {
    
    // get the values of the various buttons and selectors and store them as variables
    var gender = document.getElementById("gender").value;
    var origin = document.getElementById("origin").value;

    var count = parseInt(document.getElementById("number").value);
    var surnames = document.getElementById("surnames").checked;

    // create a request object
    var request = {
        "origin": origin,
        "gender": gender,
        "count": count,
        "surname": surnames
    };

    // turn the request json into a string
    request = JSON.stringify(request)

    console.log(request)

    // Make the http request using ajax
    const http = new XMLHttpRequest();
    const url = "/api/name";
    http.open("POST", url);
    http.send(request);

    // create a listener for when the request is returned
    http.onreadystatechange = (e) => {
        showNames(JSON.parse(http.responseText))
    }
}

// Change the page's html to show the generated names
function showNames(names) {

    // get the name area element
    var nameArea = document.getElementById("namearea");

    // delete all existing name entries
    while (nameArea.lastChild) {
        nameArea.removeChild(nameArea.lastChild);
    }

    // iterate through all of the given names
    names.names.forEach((name) => {
        
        // create the name element
        var nameElement = document.createElement("p");
        nameElement.textContent = name;
        nameElement.classList.add("name");

        // add the name element to the name area
        nameArea.appendChild(nameElement);
    });
}

function updateMap() {
    // Load the current country of origin
    var origin = document.getElementById("origin").value;

    console.log(origin)

    // get the canvas context for the worldmap
    var canvas = document.getElementById("worldmap");
    var context = canvas.getContext("2d");

    // clear the canvas
    context.clearRect(0, 0, canvas.width, canvas.height);

    // create the background image
    var background = new Image();
    background.src = "/map/background.png"

    // wait for the image to load before trying to draw it
    background.onload = function() {
        context.drawImage(background, 0, 0)

	    // only draw the country after the background has been drawn
        var country = new Image();
        country.src = "/map/" + origin + ".png"
    
        // wait for the image to load before trying to draw it
        country.onload = function() {
            context.drawImage(country, 0, 0)
        };

    };

}

// force the count to be between 1 and 99
function forceCount() {
    var count = parseInt(document.getElementById("number").value);
    if (count > 99) {
        document.getElementById("number").value = 99
    } else if (count < 1) {
        document.getElementById("number").value = 1
    }
}

// toggle the visibility of advanced options
function toggleAdvanced() {
    // get the advanced options div
    var element = document.getElementById("advancedoptions");

    // if display: none, make it visible and vice versa
    if (element.style.display == "none") {
        element.style.display = ""
    } else {
        element.style.display = "none"
    }

}

// toggle the checkbox from checked to unchecked and vice versa
function checkBox(checkbox) {
    if (checkbox.getAttribute("checked")) {
        // uncheck the box
        checkbox.removeAttribute("checked");
    } else {
        // set the box to checked
        checkbox.setAttribute("checked", true);
    }
}
